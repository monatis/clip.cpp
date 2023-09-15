import ctypes
from ctypes.util import find_library
import os
from glob import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

from huggingface_hub import model_info, snapshot_download
from huggingface_hub.utils import validate_repo_id, HFValidationError, RepositoryNotFoundError

# Note: Pass -DBUILD_SHARED_LIBS=ON to cmake to create the shared library file

cur_dir = os.getcwd()
this_dir = os.path.abspath(os.path.dirname(__file__))

# Load the shared library
ggml_lib_path, clip_lib_path = find_library("ggml"), find_library("clip")
if ggml_lib_path is None or clip_lib_path is None:
    raise RuntimeError(f"Could not find shared libraries. Please copy to the current working directory or supply the "
                       f"correct LD_LIBRARY_PATH/DYLD_LIBRARY_PATH.")

ggml_lib = ctypes.CDLL(ggml_lib_path)
clip_lib = ctypes.CDLL(clip_lib_path)


# Define the ctypes structures
class ClipTextHparams(ctypes.Structure):
    _fields_ = [
        ("n_vocab", ctypes.c_int32),
        ("num_positions", ctypes.c_int32),
        ("hidden_size", ctypes.c_int32),
        ("n_intermediate", ctypes.c_int32),
        ("projection_dim", ctypes.c_int32),
        ("n_head", ctypes.c_int32),
        ("n_layer", ctypes.c_int32),
    ]


class ClipVisionHparams(ctypes.Structure):
    _fields_ = [
        ("image_size", ctypes.c_int32),
        ("patch_size", ctypes.c_int32),
        ("hidden_size", ctypes.c_int32),
        ("n_intermediate", ctypes.c_int32),
        ("projection_dim", ctypes.c_int32),
        ("n_head", ctypes.c_int32),
        ("n_layer", ctypes.c_int32),
    ]


ClipVocabId = ctypes.c_int32
ClipVocabToken = ctypes.c_char_p
ClipVocabSpecialTokens = ctypes.c_char_p


class ClipVocab(ctypes.Structure):
    _fields_ = [
        ("token_to_id", ctypes.POINTER(ctypes.c_void_p)),
        ("id_to_token", ctypes.POINTER(ctypes.c_void_p)),
        ("special_tokens", ctypes.POINTER(ClipVocabSpecialTokens)),
    ]


class ClipTokens(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ClipVocabId)),
        ("size", ctypes.c_size_t),
    ]


class ClipImageU8(ctypes.Structure):
    _fields_ = [
        ("nx", ctypes.c_int),
        ("ny", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
    ]


class ClipImageF32(ctypes.Structure):
    _fields_ = [
        ("nx", ctypes.c_int),
        ("ny", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_float)),
    ]


class ClipContext(ctypes.Structure):
    pass


# Load the functions from the shared library
clip_model_load = clip_lib.clip_model_load
clip_model_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
clip_model_load.restype = ctypes.POINTER(ClipContext)

clip_free = clip_lib.clip_free
clip_free.argtypes = [ctypes.POINTER(ClipContext)]

clip_get_text_hparams = clip_lib.clip_get_text_hparams
clip_get_text_hparams.argtypes = [ctypes.POINTER(ClipContext)]
clip_get_text_hparams.restype = ctypes.POINTER(ClipTextHparams)

clip_get_vision_hparams = clip_lib.clip_get_vision_hparams
clip_get_vision_hparams.argtypes = [ctypes.POINTER(ClipContext)]
clip_get_vision_hparams.restype = ctypes.POINTER(ClipVisionHparams)

clip_tokenize = clip_lib.clip_tokenize
clip_tokenize.argtypes = [ctypes.POINTER(ClipContext), ctypes.c_char_p]
clip_tokenize.restype = ClipTokens

clip_image_load_from_file = clip_lib.clip_image_load_from_file
clip_image_load_from_file.argtypes = [ctypes.c_char_p, ctypes.POINTER(ClipImageU8)]
clip_image_load_from_file.restype = ctypes.c_bool

clip_image_preprocess = clip_lib.clip_image_preprocess
clip_image_preprocess.argtypes = [
    ctypes.POINTER(ClipContext),
    ctypes.POINTER(ClipImageU8),
    ctypes.POINTER(ClipImageF32),
]
clip_image_preprocess.restype = ctypes.c_bool

clip_text_encode = clip_lib.clip_text_encode
clip_text_encode.argtypes = [
    ctypes.POINTER(ClipContext),
    ctypes.c_int,
    ctypes.POINTER(ClipTokens),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_bool,
]
clip_text_encode.restype = ctypes.c_bool

clip_image_encode = clip_lib.clip_image_encode
clip_image_encode.argtypes = [
    ctypes.POINTER(ClipContext),
    ctypes.c_int,
    ctypes.POINTER(ClipImageF32),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_bool,
]
clip_image_encode.restype = ctypes.c_bool

clip_compare_text_and_image = clip_lib.clip_compare_text_and_image
clip_compare_text_and_image.argtypes = [
    ctypes.POINTER(ClipContext),
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.POINTER(ClipImageU8),
    ctypes.POINTER(ctypes.c_float),
]
clip_compare_text_and_image.restype = ctypes.c_bool

clip_similarity_score = clip_lib.clip_similarity_score
clip_similarity_score.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]
clip_similarity_score.restype = ctypes.c_float

softmax_with_sorting = clip_lib.softmax_with_sorting
softmax_with_sorting.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
]
softmax_with_sorting.restype = ctypes.c_bool

# clip_image_batch_encode = clip_lib.clip_image_batch_encode
# clip_image_batch_encode.argtypes = [
#     ctypes.POINTER(ctypes.c_void_p),
#     ctypes.c_int,
#     ctypes.POINTER(ClipImageF32),
#     ctypes.POINTER(ctypes.c_float),
# ]
# clip_image_batch_encode.restype = ctypes.c_bool

make_clip_image_u8 = clip_lib.make_clip_image_u8
make_clip_image_u8.argtypes = []
make_clip_image_u8.restype = ctypes.POINTER(ClipImageU8)

make_clip_image_f32 = clip_lib.make_clip_image_f32
make_clip_image_f32.argtypes = []
make_clip_image_f32.restype = ctypes.POINTER(ClipImageF32)


def _struct_to_dict(struct):
    return dict((field, getattr(struct, field)) for field, _ in struct._fields_)


class Clip:
    def __init__(
        self,
        model_path_or_repo_id: str,
        model_file: Optional[str] = None,
        revision: Optional[str] = None,
        verbosity: int = 0):
        """
        Loads the language model from a local file or remote repo.

        Args:
        ---
            :param model_path_or_repo_id: str
                The path to a model file  
                or the name of a Hugging Face model repo.

            :param model_file: str | None
              The name of the model file in Hugging Face repo, 
              if not specified the first bin file from the repo is choosen.

            :param revision: str | None
                The specific model version to use. It can be a branch  
                name, a tag name, or a commit id.
            :param verbosity: int { 0, 1, 2 } Default = 0
                How much verbose the model, 2 is more verbose

        """

        model_path = None
        p = Path(model_path_or_repo_id)

        if p.is_file(): 
            model_path = model_path_or_repo_id
        
        elif p.is_dir():
            model_path = self._find_model_path_from_dir(
                model_path_or_repo_id, model_file
            )
        
        else:
            try:
                validate_repo_id(model_path_or_repo_id)
                model_path = self._find_model_path_from_repo(
                    model_path_or_repo_id,
                    model_file,
                    revision=revision,
                )
            except HFValidationError as not_valid_repo_id:
                raise not_valid_repo_id
            
        self.ctx = clip_model_load(model_path.encode("utf8"), verbosity)
        self.vec_dim = self.text_config["projection_dim"]
        


    @classmethod
    def _find_model_path_from_repo(
        cls,
        repo_id: str,
        filename: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> str:
        if not filename:
            filename = cls._find_model_file_from_repo(
                repo_id=repo_id,
                revision=revision,
            )
        allow_patterns = filename or ["*.bin" ] #TODO add "*.gguf"
        path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            revision=revision,
        )
        return cls._find_model_path_from_dir(path, filename=filename)
    
    @classmethod
    def _find_model_file_from_repo(
        cls,
        repo_id: str,
        revision: Optional[str] = None,
    ) -> Optional[str]:
        try:
            repo_info = model_info(
                repo_id=repo_id,
                files_metadata=True,
                revision=revision,
            )
            files = [
                (f.size, f.rfilename)
                for f in repo_info.siblings
                #TODO add or f.rfilename.endswith(".gguf")
                if f.rfilename.endswith(".bin") 
            ]
            return min(files)[1]
        except RepositoryNotFoundError:
            raise ValueError(f"No repo found with id '{repo_id}'")

    @classmethod  
    def _find_model_path_from_dir(
        path: str,
        filename: Optional[str] = None,
        ) -> str:
            
            path = Path(path).resolve()
            if filename:
                file = path.joinpath(filename).resolve()
                if not file.is_file():
                    raise ValueError(f"Model file '{filename}' not found in '{path}'")
                return str(file)
            files = glob(path.joinpath("*.bin")) #TODO add ".gguf"
            file = min(files, key=lambda x: x[0])[1]
            return file.resolve().__str__()

    @property
    def vision_config(self) -> Dict[str, Any]:
        return _struct_to_dict(clip_get_vision_hparams(self.ctx).contents)

    @property
    def text_config(self) -> Dict[str, Any]:
        return _struct_to_dict(clip_get_text_hparams(self.ctx).contents)

    def tokenize(self, text: str) -> List[int]:
        tokens = clip_tokenize(self.ctx, text.encode("utf8"))
        return [tokens.data[i] for i in range(tokens.size)]

    def encode_text(
        self,
        tokens: List[int],
        n_threads: int = os.cpu_count(),
        normalize: bool = True,
    ) -> List[float]:
        tokens_array = (ClipVocabId * len(tokens))(*tokens)
        clip_tokens = ClipTokens(data=tokens_array, size=len(tokens))

        txt_vec = (ctypes.c_float * self.vec_dim)()

        if not clip_text_encode(
            self.ctx, n_threads, ctypes.pointer(clip_tokens), txt_vec, normalize
        ):
            raise RuntimeError("Could not encode text")

        return [txt_vec[i] for i in range(self.vec_dim)]

    def load_preprocess_encode_image(
        self, image_path: str, n_threads: int = os.cpu_count(), normalize: bool = True
    ) -> List[float]:
        image_ptr = make_clip_image_u8()
        if not clip_image_load_from_file(image_path.encode("utf8"), image_ptr):
            raise RuntimeError(f"Could not load image '{image_path}'")

        processed_image_ptr = make_clip_image_f32()
        if not clip_image_preprocess(self.ctx, image_ptr, processed_image_ptr):
            raise RuntimeError("Could not preprocess image")

        img_vec = (ctypes.c_float * self.vec_dim)()
        if not clip_image_encode(
            self.ctx, n_threads, processed_image_ptr, img_vec, normalize
        ):
            raise RuntimeError("Could not encode image")

        return [img_vec[i] for i in range(self.vec_dim)]

    def calculate_similarity(
        self, text_embedding: List[float], image_embedding: List[float]
    ) -> float:
        img_vec = (ctypes.c_float * self.vec_dim)(*image_embedding)
        txt_vec = (ctypes.c_float * self.vec_dim)(*text_embedding)

        return clip_similarity_score(txt_vec, img_vec, self.vec_dim)

    def compare_text_and_image(
        self, text: str, image_path: str, n_threads: int = os.cpu_count()
    ) -> float:
        image_ptr = make_clip_image_u8()
        if not clip_image_load_from_file(image_path.encode("utf8"), image_ptr):
            raise RuntimeError(f"Could not load image {image_path}")

        score = ctypes.c_float()
        if not clip_compare_text_and_image(
            self.ctx, n_threads, text.encode("utf8"), image_ptr, ctypes.pointer(score)
        ):
            raise RuntimeError("Could not compare text and image")

        return score.value

    def __del__(self):
        clip_free(self.ctx)
