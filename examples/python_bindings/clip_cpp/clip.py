import ctypes
import os
from typing import List, Dict, Any

# Note: Pass -DBUILD_SHARED_LIBS=ON to cmake to create the shared library file

cur_dir = os.getcwd()
this_dir = os.path.abspath(os.path.dirname(__file__))

# Load the shared library
path_to_dll = os.environ.get("CLIP_DLL", this_dir)
os.chdir(path_to_dll)
ggml_lib = ctypes.CDLL("./libggml.so")
clip_lib = ctypes.CDLL("./libclip.so")
os.chdir(cur_dir)


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


class ClipLayer(ctypes.Structure):
    _fields_ = [
        ("k_w", ctypes.POINTER(ctypes.c_void_p)),
        ("k_b", ctypes.POINTER(ctypes.c_void_p)),
        ("q_w", ctypes.POINTER(ctypes.c_void_p)),
        ("q_b", ctypes.POINTER(ctypes.c_void_p)),
        ("v_w", ctypes.POINTER(ctypes.c_void_p)),
        ("v_b", ctypes.POINTER(ctypes.c_void_p)),
        ("o_w", ctypes.POINTER(ctypes.c_void_p)),
        ("o_b", ctypes.POINTER(ctypes.c_void_p)),
        ("ln_1_w", ctypes.POINTER(ctypes.c_void_p)),
        ("ln_1_b", ctypes.POINTER(ctypes.c_void_p)),
        ("ff_i_w", ctypes.POINTER(ctypes.c_void_p)),
        ("ff_i_b", ctypes.POINTER(ctypes.c_void_p)),
        ("ff_o_w", ctypes.POINTER(ctypes.c_void_p)),
        ("ff_o_b", ctypes.POINTER(ctypes.c_void_p)),
        ("ln_2_w", ctypes.POINTER(ctypes.c_void_p)),
        ("ln_2_b", ctypes.POINTER(ctypes.c_void_p)),
    ]


class ClipTextModel(ctypes.Structure):
    _fields_ = [
        ("hparams", ClipTextHparams),
        ("token_embeddings", ctypes.POINTER(ctypes.c_void_p)),
        ("position_embeddings", ctypes.POINTER(ctypes.c_void_p)),
        ("layers", ctypes.POINTER(ClipLayer)),
        ("post_ln_w", ctypes.POINTER(ctypes.c_void_p)),
        ("post_ln_b", ctypes.POINTER(ctypes.c_void_p)),
        ("projection", ctypes.POINTER(ctypes.c_void_p)),
        ("tensors", ctypes.POINTER(ctypes.c_void_p)),
    ]


class ClipVisionModel(ctypes.Structure):
    _fields_ = [
        ("hparams", ClipVisionHparams),
        ("class_embedding", ctypes.POINTER(ctypes.c_void_p)),
        ("patch_embeddings", ctypes.POINTER(ctypes.c_void_p)),
        ("position_embeddings", ctypes.POINTER(ctypes.c_void_p)),
        ("pre_ln_w", ctypes.POINTER(ctypes.c_void_p)),
        ("pre_ln_b", ctypes.POINTER(ctypes.c_void_p)),
        ("layers", ctypes.POINTER(ClipLayer)),
        ("post_ln_w", ctypes.POINTER(ctypes.c_void_p)),
        ("post_ln_b", ctypes.POINTER(ctypes.c_void_p)),
        ("projection", ctypes.POINTER(ctypes.c_void_p)),
        ("tensors", ctypes.POINTER(ctypes.c_void_p)),
    ]


class ClipBuffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8)),
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
    _fields_ = [
        ("text_model", ClipTextModel),
        ("vision_model", ClipVisionModel),
        ("vocab", ClipVocab),
        ("use_gelu", ctypes.c_int32),
        ("ftype", ctypes.c_int32),
        ("buf_compute", ClipBuffer),
    ]


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

clip_tokenize = clip_lib.clip_tokenize_c
clip_tokenize.argtypes = [ctypes.POINTER(ClipContext), ctypes.c_char_p]
clip_tokenize.restype = ClipTokens

clip_image_load_from_file = clip_lib.clip_image_load_from_file_c
clip_image_load_from_file.argtypes = [ctypes.c_char_p, ctypes.POINTER(ClipImageU8)]
clip_image_load_from_file.restype = ctypes.c_bool

clip_image_preprocess = clip_lib.clip_image_preprocess
clip_image_preprocess.argtypes = [
    ctypes.POINTER(ClipContext),
    ctypes.POINTER(ClipImageU8),
    ctypes.POINTER(ClipImageF32),
]
clip_image_preprocess.restype = ctypes.c_bool

clip_text_encode = clip_lib.clip_text_encode_c
clip_text_encode.argtypes = [
    ctypes.POINTER(ClipContext),
    ctypes.c_int,
    ctypes.POINTER(ClipTokens),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_bool,
]
clip_text_encode.restype = ctypes.c_bool

clip_image_encode = clip_lib.clip_image_encode_c
clip_image_encode.argtypes = [
    ctypes.POINTER(ClipContext),
    ctypes.c_int,
    ctypes.POINTER(ClipImageF32),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_bool,
]
clip_image_encode.restype = ctypes.c_bool

clip_compare_text_and_image = clip_lib.clip_compare_text_and_image_c
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
    def __init__(self, model_file: str, verbosity: int = 0):
        self.ctx = clip_model_load(model_file.encode("utf8"), verbosity)
        self.vec_dim = self.text_config["projection_dim"]

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
