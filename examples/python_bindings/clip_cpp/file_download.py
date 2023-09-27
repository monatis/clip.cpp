import json
import sys
import urllib.request
from pathlib import Path
from typing import Optional, List, Dict, TypedDict, Union

from .exceptions import RepositoryNotFoundError, RepositoryFileNameNotFound


class Model:
    def __init__(
        self,
        _id: str,
        id: str,
        likes: int,
        private: bool,
        downloads: int,
        tags: List[str],
        modelId: str,
    ):
        self._id = _id
        self.id = id
        self.likes = likes
        self.private = private
        self.downloads = downloads
        self.tags = tags
        self.modelId = modelId

    def __str__(self):
        return f"repo_id: {self.modelId}"


class BlobLfsInfo(TypedDict, total=False):
    size: int
    sha256: str
    pointer_size: int


class RepoFile:
    def __init__(
        self,
        rfilename: str,
        size: Optional[int] = None,
        blobId: Optional[str] = None,
        lfs: Optional[BlobLfsInfo] = None,
        **kwargs,
    ):
        self.rfilename = rfilename  # filename relative to the repo root
        # Optional file metadata
        self.size = size
        self.blob_id = blobId
        self.lfs = lfs

        for k, v in kwargs.items():
            setattr(self, k, v)


class ModelInfo:
    def __init__(
        self,
        *,
        modelId: Optional[str] = None,
        sha: Optional[str] = None,
        lastModified: Optional[str] = None,
        tags: Optional[List[str]] = None,
        pipeline_tag: Optional[str] = None,
        siblings: Optional[List[Dict]] = None,
        private: bool = False,
        author: Optional[str] = None,
        config: Optional[Dict] = None,
        securityStatus: Optional[Dict] = None,
        **kwargs,
    ):
        self.modelId = modelId
        self.sha = sha
        self.lastModified = lastModified
        self.tags = tags
        self.pipeline_tag = pipeline_tag
        self.siblings = (
            [RepoFile(**x) for x in siblings] if siblings is not None else []
        )
        self.private = private
        self.author = author
        self.config = config
        self.securityStatus = securityStatus
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        r = f"Model Name: {self.modelId}, Tags: {self.tags}"
        if self.pipeline_tag:
            r += f", Task: {self.pipeline_tag}"
        return r


def model_info(
    repo_id: str,
    files_metadata: bool = False,
) -> ModelInfo:
    get_files_metadata = (
        urllib.parse.urlencode({"blobs": files_metadata}) if files_metadata else ""
    )
    url = f"https://huggingface.co/api/models/{repo_id}/?" + get_files_metadata
    try:
        response = urllib.request.urlopen(url)
        data = json.load(response)
        return ModelInfo(**data)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise RepositoryNotFoundError
        elif e.peek().decode("utf-8", errors="ignore") == "Entry not found":
            raise RepositoryFileNameNotFound
        else:
            print(f"\nError getting Info about the repo_id: {e}")

    except Exception as e:
        print(f"\nError getting Info about the repo_id: {e}")


def model_download(repo_id: str, file_name: Union[str, List[str]]) -> str:
    """Download HF model and returns the Path of the Downloaded file"""

    # create a model directory to save the files neatly in one folder
    models_dir = Path("./models/")
    models_dir.mkdir(exist_ok=True)
    destination_path = models_dir.joinpath(file_name)

    url = f"https://huggingface.co/{repo_id}/resolve/main/{file_name}"

    def reporthook(count, block_size, total_size):
        # Calculate the progress
        downloaded_chunk = count * block_size
        progress = (downloaded_chunk / total_size) * 100

        # print(downloaded_chunk // total_size)
        bar = "".join(["=" if i <= progress / 2 else " " for i in range(50)])
        sys.stdout.write(
            f"\r[{bar}] {progress:.1f}% ({downloaded_chunk/1024**2:.2f} MB/{total_size/1024**2:.0f} MB)"
        )
        sys.stdout.flush()

    try:
        print(f"[File Info] {destination_path}")
        # check if the file exists and matches the size of that in the network
        network_file_size = urllib.request.urlopen(url).info().get("Content-Length", 0)
        if destination_path.is_file() and destination_path.stat().st_size == int(
            network_file_size
        ):
            # raise FileNameAlreadyExists
            return models_dir

        urllib.request.urlretrieve(url, destination_path, reporthook=reporthook)
        sys.stdout.write("\n")
        print(f"File downloaded to {destination_path}")
        return models_dir

    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise RepositoryNotFoundError
        elif e.peek().decode("utf-8", errors="ignore") == "Entry not found":
            raise RepositoryFileNameNotFound
        else:
            print(f"\nError downloading file: {e}")

    except Exception as e:
        print(f"\nError downloading file: {e}")


def get_models() -> List[Model]:
    url = f"https://huggingface.co/api/models?filter=clip-cpp-gguf"
    try:
        response = urllib.request.urlopen(url)
        data = json.load(response)
        return [Model(**item) for item in data]
    except urllib.error.HTTPError as e:
        print(f"\nError listing available models: {e}")

    except Exception as e:
        print(f"\nError listing available models: {e}")


def available_models():
    if len(sys.argv) > 1 and sys.argv[-1] != "clip-cpp-models":
        repo_id = sys.argv[-1]
        repo = model_info(repo_id=repo_id, files_metadata=True)
        print(f"Available models in repo {repo_id}:")
        for model in repo.siblings:
            if model.rfilename.endswith(".gguf"):
                name = model.rfilename
                size = model.size / (1024 * 1024)
                print(f"    model: {name} ({size:.2f} MB)")

        return

    models = get_models()
    print(
        "Below are available models on HuggingFace Hub that you can use with the clip-cpp package.\n"
    )
    print("Available models:")
    for model in models:
        print(model)

    print(
        "\nYou can pass one of the repo IDs above directly to `Clip()`, and it will download the smallest model in that repo automatically."
    )
    print(
        "Alternatively, you can choose which to load from that repo by passing a value to `model_file` argument."
    )
    print("\nTo see model files in a repo, type `clip-cpp-models <repo_id>`")
