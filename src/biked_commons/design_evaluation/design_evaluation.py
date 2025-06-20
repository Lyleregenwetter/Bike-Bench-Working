from abc import abstractmethod, ABC
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import dill
from PIL import Image

from biked_commons.embedding import clip_embedding_calculator
from biked_commons.transformation import interface_points, framed, ordered_columns
from biked_commons.ergonomics import joint_angles
from biked_commons.prediction import aero_predictor, clip_predictor
from biked_commons.prediction.prediction_utils import Preprocessor
from biked_commons.resource_utils import models_and_scalers_path
from biked_commons.data_loading import data_loading
from biked_commons.validation.base_validation_function import construct_tensor_validator
from biked_commons.validation.bike_bench_validation_functions import bike_bench_validation_functions





class EvaluationFunction(ABC):
    def __init__(self, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def variable_names(self) -> List[str]:
        pass

    @abstractmethod
    def return_names(self) -> List[str]:
        pass

    @abstractmethod # 1 = objective, 0 = constraint
    def return_types(self) -> List[str]:
        pass

    @abstractmethod
    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        pass


class AeroEvaluator(EvaluationFunction):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__(device, dtype)
        model_path = models_and_scalers_path("aero_model.pt")
        scaler_path = models_and_scalers_path("aero_scaler.pt")
        self.model = torch.load(model_path, weights_only=False).to(self.device)
        self.preprocessor = Preprocessor(scaler_path=scaler_path, preprocess_fn=aero_predictor.calculate_features, device=device)

    def variable_names(self) -> List[str]:
        return [
            "Stack",
            "Handlebar style OHCLASS: 0", "Handlebar style OHCLASS: 1", "Handlebar style OHCLASS: 2",
            "Seat angle", "Saddle height", "Head tube length textfield", "Head tube lower extension2",
            "Head angle", "DT Length"
        ]

    def return_names(self) -> List[str]:
        return ['Drag Force']
    
    def return_types(self) -> List[str]:
        return [1]

    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        int_pts = interface_points.calculate_interface_points(designs)
        assert "Rider" in conditioning, "Rider dimensions must be provided in conditioning to calculate aerodynamics."
        rider_dims = conditioning["Rider"]
        #if rider_dims is a 1D tensor, expand it to match the batch size of designs
        if rider_dims.dim() == 1:
            rider_dims = rider_dims.unsqueeze(0).expand(designs.shape[0], -1)
        elif rider_dims.shape[0] == 1:
            rider_dims = rider_dims.expand(designs.shape[0], -1)
        rider_dims = rider_dims.to(self.device, dtype=self.dtype)
        combinations = torch.cat((int_pts, rider_dims), dim=1)
        combinations = self.preprocessor(combinations)
        predictions = self.model(combinations)
        predictions = torch.clip(predictions, min=0)
        return predictions

class FrameValidityEvaluator(EvaluationFunction):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__(device, dtype)
        model_path = models_and_scalers_path("validity_model.pt")
        scaler_path = models_and_scalers_path("validity_scaler.pt")
        self.model = torch.load(model_path, weights_only=False).to(self.device)
        self.preprocessor = Preprocessor(scaler_path=scaler_path, preprocess_fn=None, device=device)
        
        self.converter = framed.clip_to_framed_tensor_builder(ordered_columns.bike_bench_columns, framed.FRAMED_ORDERED_COLUMNS)

    def variable_names(self) -> List[str]:
        return ordered_columns.bike_bench_columns

    def return_names(self) -> List[str]:
        return ['Predicted Frame Validity']
    
    def return_types(self) -> List[str]:
        return [0]

    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:

        framed_tensor = self.converter(designs)
        framed_tensor = framed_tensor.to(self.device, dtype=self.dtype)
        framed_tensor = self.preprocessor(framed_tensor)
        predictions = self.model(framed_tensor)
        validity = predictions-0.5
        return validity
    
class StructuralEvaluator(EvaluationFunction):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__(device, dtype)
        model_path = models_and_scalers_path("structural_model.pt")
        scaler_path = models_and_scalers_path("structural_scaler.pt")
        self.model = torch.load(model_path, weights_only=False).to(self.device)
        self.preprocessor = Preprocessor(scaler_path=scaler_path, preprocess_fn=None, device=device)
        
        self.converter = framed.clip_to_framed_tensor_builder(ordered_columns.bike_bench_columns, framed.FRAMED_ORDERED_COLUMNS)

    def variable_names(self) -> List[str]:
        return ordered_columns.bike_bench_columns

    def return_names(self) -> List[str]:
        return ['Mass', 'Planar Compliance', 'Transverse Compliance', 'Eccentric Compliance', 'Planar Safety Factor', 'Eccentric Safety Factor']

    def return_types(self) -> List[str]:
        return [1,1,1,1,0,0]

    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        framed_tensor = self.converter(designs)
        framed_tensor = framed_tensor.to(self.device, dtype=self.dtype)
        framed_tensor = self.preprocessor(framed_tensor)
        predictions = self.model(framed_tensor)
        predictions = torch.clip(predictions, min=0)
        predictions[:, 4:6] = 1.5 - predictions[:, 4:6]
        return predictions

class AestheticsEvaluator(EvaluationFunction):
    def __init__(self,
                 mode: str = "Image",
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 batch_size: int = None):
        super().__init__(device, dtype)
        model_path  = models_and_scalers_path("clip_model.pt")
        scaler_path = models_and_scalers_path("clip_scaler.pt")
        self.preprocessor = Preprocessor(
            scaler_path=scaler_path,
            preprocess_fn=clip_predictor.remove_wall_thickness,
            device=device
        )
        self.model = torch.load(model_path, weights_only=False).to(self.device)
        self.model.eval()

        self.mode = mode  # "Image", "Image Path", or "Text"
        self.embedding_model = clip_embedding_calculator.ClipEmbeddingCalculator(
            device=self.device,
            batch_size=batch_size
        )

    def variable_names(self) -> List[str]:
        return ordered_columns.bike_bench_columns

    def return_names(self) -> List[str]:
        if self.mode in ["Image", "Image Path"]:
            return ["Cosine Distance to Image"]
        elif self.mode == "Text":
            return ["Cosine Distance to Text"]
        elif self.mode == "Embedding":
            return ["Cosine Distance to Embedding"]

    def return_types(self) -> List[str]:
        return [1]

    def evaluate(self,
                 designs: torch.Tensor,
                 conditioning: dict = {}) -> torch.Tensor:
        cond = conditioning.get(self.mode)
        if cond is None:
            raise ValueError(f"No conditioning provided for mode '{self.mode}'")

        # Prepare a list of items for embedding
        if self.mode == "Image":
            if isinstance(cond, torch.Tensor):
                items = [cond]
            elif isinstance(cond, (list, tuple)):
                items = list(cond)
            else:
                raise TypeError("For Image mode, conditioning must be a Tensor or list of Tensors")
            embed = self.embedding_model.embed_images(items)

        elif self.mode == "Image Path":
            if isinstance(cond, str):
                paths = [cond]
            elif isinstance(cond, (list, tuple)):
                paths = list(cond)
            else:
                raise TypeError("For Image Path mode, conditioning must be a path or list of paths")
            imgs = [Image.open(p) for p in paths]
            embed = self.embedding_model.embed_images(imgs)

        elif self.mode == "Text":
            if isinstance(cond, str):
                texts = [cond]
            elif isinstance(cond, (list, tuple)):
                texts = list(cond)
            else:
                raise TypeError("For Text mode, conditioning must be text or list of texts")
            embed = self.embedding_model.embed_texts(texts)
        elif self.mode == "Embedding":
            if isinstance(cond, torch.Tensor):
                embed = cond
            else:
                raise TypeError("For Embedding mode, conditioning must be a Tensor ")
            
            if embed.dim() == 1:
                embed = embed.unsqueeze(0).expand(designs.shape[0], -1)
            elif embed.shape[0] == 1:
                embed = embed.expand(designs.shape[0], -1)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        
        designs = self.preprocessor(designs)
        preds   = self.model(designs)
        N       = preds.size(0)
        B_cond  = embed.size(0)
        if B_cond == 1 and N > 1:
            embed = embed.expand(N, -1)
        elif B_cond != N:
            raise ValueError(
                f"Number of condition embeddings ({B_cond}) "
                f"does not match number of designs ({N})"
            )
        
        embed = embed.to(self.device, dtype=self.dtype)

        cos_sim = F.cosine_similarity(preds, embed, dim=1)
        return (1 - cos_sim) / 2

class ValidationEvaluator(EvaluationFunction):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__(device, dtype)
        self.clip_parameters = data_loading.load_bike_bench_train().columns.tolist() #TODO maybe include a list somewhere to avoid loading a dataset?
        validator, validation_names = construct_tensor_validator(bike_bench_validation_functions, self.clip_parameters)
        self.validator = validator
        self.validation_names = validation_names

    def variable_names(self) -> List[str]:
        return self.clip_parameters

    def return_names(self) -> List[str]:
        return self.validation_names
    
    def return_types(self) -> List[str]:
        return [0] * len(self.validation_names)

    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        # designs = designs.to(self.device, dtype=self.dtype)
        predictions = self.validator(designs)
        return predictions

class ErgonomicsEvaluator(EvaluationFunction):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__(device, dtype)
    def variable_names(self) -> List[str]:
        return [
            "Stack",
            "Handlebar style OHCLASS: 0", "Handlebar style OHCLASS: 1", "Handlebar style OHCLASS: 2",
            "Seat angle", "Saddle height", "Head tube length textfield", "Head tube lower extension2",
            "Head angle", "DT Length"
        ]

    def return_names(self) -> List[str]:
        return ['Knee Angle Error', 'Hip Angle Error', "Arm Angle Error"]
    
    def return_types(self) -> List[str]:
        return [1, 1, 1]

    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        assert "Rider" in conditioning, "Rider dimensions must be provided in conditioning to calculate ergonomics."
        rider_dims = conditioning["Rider"]
        if rider_dims.dim() == 1:
            rider_dims = rider_dims.unsqueeze(0).expand(designs.shape[0], -1)

        rider_dims = rider_dims.to(self.device, dtype=self.dtype)

        assert "Use Case" in conditioning, "Use Case must be provided in conditioning to calculate ergonomics."
        use_case = conditioning["Use Case"]
        if not isinstance(use_case, torch.Tensor):
            raise TypeError(f"Use Case must be a torch.Tensor, got {type(use_case)}")

        if use_case.dim() == 1:
            # single one-hot of shape (3,)
            if use_case.shape != (3,):
                raise ValueError(f"If 1D, Use Case tensor must have shape (3,), got {tuple(use_case.shape)}")
            # must be exactly 0s and 1s
            if not torch.logical_or(use_case == 0, use_case == 1).all():
                raise ValueError("Use Case 1D tensor must contain only 0s and 1s")
            # must sum to 1
            if use_case.sum().item() != 1:
                raise ValueError("Use Case 1D tensor must be a valid one-hot vector (sum == 1)")
            # broadcast to (n,3)
            n = designs.shape[0]
            use_case = use_case.unsqueeze(0).repeat(n, 1)

        elif use_case.dim() == 2:
            # batch of one-hots, shape (n,3)
            n, k = use_case.shape
            if k != 3:
                raise ValueError(f"If 2D, Use Case tensor must have shape (n,3), got {tuple(use_case.shape)}")
            if n == 1:
                use_case = use_case.expand(designs.shape[0], -1)
            # check binary values
            if not torch.logical_or(use_case == 0, use_case == 1).all():
                raise ValueError("Use Case 2D tensor must contain only 0s and 1s")
            # each row sums to exactly 1
            row_sums = use_case.sum(dim=1)
            bad_rows = (row_sums != 1).nonzero(as_tuple=False).flatten()
            if bad_rows.numel() > 0:
                raise ValueError(f"Rows at indices {bad_rows.tolist()} are not valid one-hot vectors")

        else:
            raise ValueError(f"Use Case tensor must be 1D or 2D, got {use_case.dim()}D")
        
        index_to_label = ["road", "mtb", "commute"]
        use_case_list = [index_to_label[idx] for idx in use_case.argmax(axis=1)]

        int_pts = interface_points.calculate_interface_points(designs)
        predictions = joint_angles.dist_to_1SD(int_pts, rider_dims, use_case_list)
        return predictions


class UsabilityEvaluator(EvaluationFunction):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__(device, dtype)
        scaler_path = models_and_scalers_path("usability_scaler.pt")
        model_path = models_and_scalers_path("usability_model.pt")
        self.model = torch.load(model_path, weights_only=False).to(self.device)
        self.preprocessor = Preprocessor(scaler_path=scaler_path, preprocess_fn=None, device=device)

    def variable_names(self) -> List[str]:
        return ordered_columns.USABILITY_COLUMNS

    def return_names(self) -> List[str]:
        return ['Usability Score - 0 to 1']

        
    def return_types(self) -> List[str]:
        return [1]

    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        designs = self.preprocessor(designs)
        predictions = 1- self.model(designs)
        return torch.clip(predictions, min=0, max=1)


class PreprocessingFunction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def variable_names(self) -> List[str]:
        pass

    @abstractmethod
    def process(self, designs: torch.Tensor) -> torch.Tensor:
        pass
    

class clip_bools_to_0_1(PreprocessingFunction):
    def __init__(self, device="cpu"):
        self.device = device

    def variable_names(self) -> List[str]:
        return ordered_columns.oh_bool_columns

    def process(self, designs: torch.Tensor) -> torch.Tensor:
        # Clip the values to be between 0 and 1
        designs = torch.clip(designs, min=0, max=1)
        return designs
    
class normalize_onehot(PreprocessingFunction):
    def __init__(self, device="cpu"):
        self.device = device

        # Define groups of one-hot column names
        self.groups = ordered_columns.oh_columns

        # Flatten all variable names for indexing
        self._variable_names = [col for group in self.groups for col in group]

    def variable_names(self) -> List[str]:
        return self._variable_names

    def process(self, designs: torch.Tensor) -> torch.Tensor:
        # List to hold processed group slices
        normalized_groups = []

        current_col = 0
        for group in self.groups:
            num_cols = len(group)
            group_slice = designs[:, current_col:current_col + num_cols]
            group_sum = group_slice.sum(dim=1, keepdim=True).clamp(min=1e-8)
            normalized_group = group_slice / group_sum
            normalized_groups.append(normalized_group)
            current_col += num_cols

        # Concatenate normalized groups along feature dimension
        return torch.cat(normalized_groups, dim=1)

def get_standard_preprocessing(device):
    return [clip_bools_to_0_1(device=device),
            normalize_onehot(device=device)]


    
def construct_tensor_evaluator(evaluation_functions: List[EvaluationFunction], column_names: List[str], preprocessing_fn_set = get_standard_preprocessing, device="cpu"):
    preprocessing_fns = preprocessing_fn_set(device)

    column_names = list(column_names)

    # Flatten all return names across evaluators
    all_return_names = []
    all_return_types = []
    for evaluation_function in evaluation_functions:
        all_return_names.extend(evaluation_function.return_names())
        all_return_types.extend(evaluation_function.return_types())

    def evaluate_tensor(designs: torch.Tensor, conditioning={}) -> torch.Tensor:
        n = designs.shape[0]
        total_outputs = sum(len(evaluation_function.return_names()) for evaluation_function in evaluation_functions)
        results_tensor = torch.zeros((n, total_outputs), dtype=torch.float32, device=designs.device)

        current_col = 0
        for preprocessing_fn in preprocessing_fns:
            var_indices = [column_names.index(var) for var in preprocessing_fn.variable_names()]
            sliced_designs = designs[:, var_indices]
            processed = preprocessing_fn.process(sliced_designs)

            updated = designs.clone()
            updated[:, var_indices] = processed
            designs = updated

        for evaluation_function in evaluation_functions:
            var_indices = [column_names.index(var) for var in evaluation_function.variable_names()]
            sliced_designs = designs[:, var_indices]

            res = evaluation_function.evaluate(sliced_designs, conditioning)  # Expect shape (n,) or (n, k)

            if res.dim() == 1:
                res = res.unsqueeze(1)

            num_outputs = res.shape[1]
            results_tensor[:, current_col:current_col + num_outputs] = res
            current_col += num_outputs

        return results_tensor

    return evaluate_tensor, all_return_names, all_return_types

def construct_dataframe_evaluator(evaluation_functions: List[EvaluationFunction]):

    def evaluate_dataframe(designs: pd.DataFrame, conditioning={}) -> pd.DataFrame:
        designs_tensor = torch.tensor(designs.values, dtype=torch.float32)
        tensor_evaluator, return_names, return_types = construct_tensor_evaluator(evaluation_functions, list(designs.columns))
        results_tensor = tensor_evaluator(designs_tensor, conditioning)

        results_df = pd.DataFrame(
            results_tensor.detach().cpu().numpy(),
            columns=return_names,
            index=designs.index
        )

        return results_df, return_types

    return evaluate_dataframe





def get_standard_evaluations(device, aesthetics_mode = "Embedding") -> List[EvaluationFunction]:

    StandardEvaluations = [
        UsabilityEvaluator(device=device),
        AeroEvaluator(device=device),
        ErgonomicsEvaluator(device=device),
        AestheticsEvaluator(mode=aesthetics_mode, batch_size=64, device=device),
        StructuralEvaluator(device=device),
        ValidationEvaluator(device=device),
        FrameValidityEvaluator(device=device)
    ]

    return StandardEvaluations