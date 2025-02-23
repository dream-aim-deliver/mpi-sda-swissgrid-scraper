import logging
import tempfile
from typing import List
import requests
from app.config import SUPPORTED_DATASET_EVALSCRIPTS
from app.sdk.models import (
    BaseJobState,
    JobOutput,
    KernelPlancksterSourceData,
    ProtocolEnum,
)
from app.sdk.scraped_data_repository import ScrapedDataRepository
from app.time_travel.models import Error, Image, KeyFrame, Metadata, SwissgridRowSchema
from utils import parse_relative_path


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def __filter_paths_by_timestamp(
    timestamp: str, relative_paths: List[KernelPlancksterSourceData]
) -> List[str]:
    return [
        path.relative_path for path in relative_paths if timestamp in path.relative_path
    ]


def generate_time_travel_metadata(
    job_id: int,
    tracer_id: str,
    scraped_data_repository: ScrapedDataRepository,
    relevant_source_data: list[KernelPlancksterSourceData],
    protocol: ProtocolEnum,
    predict_url: str,
    prediction_model_name: str,
    power_plant_name: str,
    power_plant_bounding_box: str,
) -> JobOutput:

    case_study_name = "swissgrid"
    failed = False
    timestamps: List[str] = []
    relative_paths_for_agent: List[str] = []

    IMAGE_SEQUENCE = [
        "thermal",
        "natural",
        "optical-thickness",
        "moisture",
        "chlorophyll",
    ]

    for source_data in relevant_source_data:
        relative_path = source_data.relative_path
        (
            _,
            _,
            _,
            timestamp,
            _,
            _,
            _,
            file_extension,
        ) = parse_relative_path(relative_path=relative_path)

        timestamps.append(timestamp)

        if file_extension in ["json", "csv", "txt"]:
            relative_paths_for_agent.append(relative_path)

    timestamps = list(set(timestamps))

    metadata: Metadata = Metadata(
        caseStudy="swissgrid",
        imageKinds=[],
        relativePathsForAgent=relative_paths_for_agent,
        keyframes=[],
    )

    for timestamp in timestamps:
        keyframe = KeyFrame(
            timestamp=timestamp,
            images=[],
            data=[],
            dataDescription=f"This data is a collection of predictions at certain timestamps for indicating whether the power plant {power_plant_name} is on/off.",
        )

        timestamp_relative_paths = __filter_paths_by_timestamp(
            timestamp, relevant_source_data
        )

        images_paths = [
            path
            for path in timestamp_relative_paths
            if path.endswith((".png", ".jpg", ".jpeg"))
        ]
        sorted_image_paths = []
        for image_path in images_paths:
            (
                _,
                _,
                _,
                _,
                _,
                evalscript_name,
                _,
                _,
            ) = parse_relative_path(relative_path=image_path)
            try:
                index = IMAGE_SEQUENCE.index(evalscript_name)
            except ValueError:
                keyframe.images.append(
                    Error(
                        errorMessage=f"Evalscript {evalscript_name} is not valid for Swissgrid Case Study.",
                        errorName="UnsupportedEvalscript",
                    )
                )
            sorted_image_paths.append((image_path, index))

        sorted_image_paths = sorted(sorted_image_paths, key=lambda x: x[1])
        images_paths = [x[0] for x in sorted_image_paths]

        for image_path in images_paths:
            (
                _,
                _,
                _,
                timestamp,
                dataset,
                evalscript_name,
                hash,
                file_extension,
            ) = parse_relative_path(relative_path=image_path)

            if dataset not in SUPPORTED_DATASET_EVALSCRIPTS:
                keyframe.images.append(
                    Error(
                        errorMessage=f"Dataset {dataset} is not supported",
                        errorName="UnsupportedDataset",
                    )
                )
                continue

            supported_eval_scripts = [
                x["name"]
                for x in SUPPORTED_DATASET_EVALSCRIPTS[dataset]["supported_evalscripts"]
            ]

            if evalscript_name not in supported_eval_scripts:
                keyframe.images.append(
                    Error(
                        errorMessage=f"Evalscript {evalscript_name} is not supported for {dataset}.",
                        errorName="UnsupportedEvalscript",
                    )
                )
                continue

            if evalscript_name not in metadata.imageKinds:
                metadata.imageKinds.append(evalscript_name)

            evalscript = next(
                (
                    x
                    for x in SUPPORTED_DATASET_EVALSCRIPTS[dataset][
                        "supported_evalscripts"
                    ]
                    if x["name"] == evalscript_name
                ),
                None,
            )

            if not evalscript:
                keyframe.images.append(
                    Error(
                        errorMessage=f"Evalscript {evalscript_name} not found for {dataset}.",
                        errorName="MissingEvalscript",
                    )
                )
                continue
            
            if hash == "empty":
                keyframe.images.append(Error(
                    errorMessage=f"No Satellite Image was found for this timestamp. Possibly the satellite did not pass over the given coordinates.",
                    errorName="EmptyImage",
                ))
                continue

            img_to_append = Image(
                relativePath=image_path,
                kind=evalscript_name,
                description=f"dataset: {dataset} | power plant: {power_plant_name} | coords: {power_plant_bounding_box} | details: {evalscript['description']}",  # TODO: check if the scraper allows for these details
            )

            keyframe.images.append(img_to_append)

            # Make prediction for the given timestamp
            try:
                response = requests.post(
                    predict_url,
                    json={
                        "relative_paths": images_paths,
                        "model_name": prediction_model_name,
                    },
                )
                if response.status_code != 200:
                    keyframe.data.append(
                        Error(
                            errorName="PredictionError",
                            errorMessage=f"Error while making prediction: Status Code: {response.status_code}. Data {response.text}",
                        )
                    )
                    continue

                try:
                    prediction_result = response.json()
                except Exception as e:
                    keyframe.data.append(
                        Error(
                            errorName="PredictionError",
                            errorMessage=f"Could not decode JSON response from the predictor service. {e}",
                        )
                    )
                    continue
  
                for data_row in prediction_result:
                    keyframe.data.append(
                        SwissgridRowSchema(
                            timestamp=timestamp,
                            model=data_row["label"],
                            prediction=data_row["prediction"],
                            confidence=data_row["confidence"],
                        )
                    )

            except Exception as e:
                keyframe.data.append(
                    Error(
                        errorName="PredictionError",
                        errorMessage=f"Error while making prediction: {e}",
                    )
                )
            metadata.keyframes.append(keyframe)


    with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as out:
        with open(out.name, "w") as f:
            f.write(metadata.model_dump_json(indent=2))

        relative_path = f"{case_study_name}/{tracer_id}/{job_id}/metadata.json"
        out_source_data = KernelPlancksterSourceData(
            name="time_travel_metadata.json",
            protocol=protocol,
            relative_path=relative_path,
        )

        try:
            scraped_data_repository.register_scraped_json(
                job_id=job_id,
                source_data=out_source_data,
                local_file_name=out.name,
            )

        except Exception as e:
            logger.error(f"Failed to upload time travel metadata: {e}")
            failed = True

    if failed:
        return JobOutput(
            job_state=BaseJobState.FAILED,
            tracer_id=tracer_id,
            source_data_list=[],
        )

    return JobOutput(
        job_state=BaseJobState.FINISHED,
        tracer_id=tracer_id,
        source_data_list=[out_source_data],
    )
