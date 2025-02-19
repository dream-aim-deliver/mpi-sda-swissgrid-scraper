from datetime import datetime
from app.sdk.models import KernelPlancksterSourceData, BaseJobState, JobOutput
from app.sdk.scraped_data_repository import KernelPlancksterSourceData, ScrapedDataRepository
import time
import numpy as np
from typing import List
from PIL import Image
import logging
import os
#import io
from PIL import Image
import pandas as pd
from app.utils import generate_relative_path 
from app.sentinel_scraper import get_satellite_data

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_image(latitude: str, longitude: str, date:datetime, resolution: int, image_dir: str, data_type: str, sentinel_client_secret: str, sentinel_client_id: str) -> Image.Image | None:
    latitude = float(latitude)
    longitude = float(longitude)
    buffer = 0.02 # Small buffer for bounding box
    coords = (longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer)
    try:
        image = get_satellite_data(coords, date, resolution, image_dir, data_type, sentinel_client_secret, sentinel_client_id)
        #image = Image.open(io.BytesIO(image))
        return image
    except Exception as e:
        logger.warning(f"Failed to retrieve {data_type} image for {date} due to error: {e}")
        return None


def save_image(image, path, factor=1.0, clip_range=(0, 1)):
    """Save the image to the given path."""
    np_image = np.array(image) * factor
    np_image = np.clip(np_image, clip_range[0], clip_range[1])
    np_image = (np_image * 255).astype(np.uint8)
    Image.fromarray(np_image).save(path)


def scrape(
        case_study_name: str,
        job_id: int,
        tracer_id: str,
        scraped_data_repository: ScrapedDataRepository,
        log_level: str,
        latitude:str,
        longitude:str,
        start_date: datetime,
        end_date: datetime,
        resolution:int,
        data_type:str,
        file_dir: str,
        sentinel_client_id:str,
        sentinel_client_secret:str,
        predict_url: str,
        prediction_model_name: str
    ) -> JobOutput:
    
    job_state = BaseJobState.CREATED

    start_time = time.time()
    image_path = None
    relative_path = None
    success = False
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    try:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=log_level)

        protocol = scraped_data_repository.protocol

        output_data_list: List[KernelPlancksterSourceData] = []

        logger.info(f"{job_id}: Starting Job")

        job_state = BaseJobState.RUNNING
        image_dir = os.path.join(file_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        for date in dates:
            image = fetch_image(latitude, longitude, date, resolution, image_dir, data_type, sentinel_client_id, sentinel_client_secret)
            
            unix_timestamp = int(date.timestamp())

            if image is None:
                relative_path = None
                raise Exception(f"Image for {date} could not be fetched")

            file_extension = image.format.lower() if image.format else "png"
            image_filename = f"Swissgrid_test.{file_extension}"
            image_path = os.path.join(image_dir, "scraped", image_filename)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            save_image(image, image_path, factor=1.5 / 255, clip_range=(0, 1))
            logger.info(f"Scraped Image at {time.time()} and saved to: {image_path}")
            relative_path = generate_relative_path(
                case_study_name=case_study_name,
                tracer_id=tracer_id,
                job_id=job_id,
                timestamp=unix_timestamp,
                dataset="sentinel",     
                evalscript_name=data_type,
                image_hash="nohash",
                file_extension=file_extension
            )

            media_data = KernelPlancksterSourceData(
            name="swissgrid", #dummy name
            protocol=protocol,
            relative_path=relative_path,
            )

            scraped_data_repository.register_scraped_photo(
                job_id=job_id,
                source_data=media_data,
                local_file_name=image_path,
            )

            output_data_list.append(media_data)
            success = True

    except Exception as error:
        logger.error(f"{job_id}: Unable to scrape data. Job with tracer_id {tracer_id} failed. Job state was '{job_state.value}' Error: {error}")

    finally:
        if image_path:
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    logger.info(f"Deleted scraped image at {image_path}")
                except Exception as e:
                    logger.warning(f"Could not delete scraped image: {e}")

        response_time = time.time() - start_time
        if success:
            logger.info(f"{job_id}: Job finished successfully. Response time: {response_time:.2f} seconds")
            return JobOutput(
            job_state=BaseJobState.FINISHED,
            tracer_id=tracer_id,
            source_data_list=output_data_list
            )
