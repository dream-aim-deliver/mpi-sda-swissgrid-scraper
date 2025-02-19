from typing import NamedTuple


class KernelPlancksterRelativePath(NamedTuple):
    case_study_name: str
    tracer_id: str
    job_id: str
    timestamp: str
    dataset: str
    evalscript_name: str
    image_hash: str
    file_extension: str


def generate_relative_path(case_study_name, tracer_id, job_id, timestamp, dataset, evalscript_name, image_hash, file_extension):
    return f"{case_study_name}/{tracer_id}/{job_id}/{timestamp}/sentinel/{dataset}_{evalscript_name}_{image_hash}.{file_extension}"

def parse_relative_path(relative_path) -> KernelPlancksterRelativePath:
    parts = relative_path.split("/")
    case_study_name = parts[0]
    tracer_id = parts[1]
    job_id = parts[2]
    timestamp = parts[3]
    dataset, evalscript_name, image_hash_extension = parts[5].split("_")
    image_hash, file_extension = image_hash_extension.split(".")
    return KernelPlancksterRelativePath(case_study_name, tracer_id, job_id, timestamp, dataset, evalscript_name, image_hash, file_extension)

