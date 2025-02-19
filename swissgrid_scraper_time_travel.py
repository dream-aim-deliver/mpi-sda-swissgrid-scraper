import json
import logging
import sys
from app.config import SUPPORTED_DATASET_EVALSCRIPTS
from app.sdk.scraped_data_repository import ScrapedDataRepository
from app.setup import datetime_parser, setup, string_validator
from app.scraper import scrape

def main(
    job_id: int,
    tracer_id: str,
    latitude: str,
    longitude: str,
    start_date: str,
    end_date: str,
    resolution: int,
    data_type: str,
    file_dir: str,
    sentinel_client_id:str,
    sentinel_client_secret:str,
    predict_url: str,
    prediction_model_name: str,
    datasets_evalscripts: str,
    kp_host: str,
    kp_port: str,
    kp_auth_token: str,
    kp_scheme: str,
    log_level: str = "WARNING"
) -> None:

    try:
        datasets_evalscripts_loaded = json.loads(datasets_evalscripts)
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

        case_study_name = "swissgrid"

        required_variables = [
            case_study_name,
            job_id,
            tracer_id,
            latitude,
            longitude,
            sentinel_client_id,
            sentinel_client_secret,
            predict_url,
            prediction_model_name
        ] 
    
        if not all(required_variables):
            raise ValueError(f"The following variables are required: {",".join(required_variables)}")

        string_variables = {
            "case_study_name": case_study_name,
            "job_id": job_id,
            "tracer_id": tracer_id,
            "latitude": latitude,
            "longitude": longitude,
        }

        logger.info(f"Validating string variables:  {string_variables}")

        for name, value in string_variables.items():
            string_validator(f"{value}", name)

        logger.info(f"String variables validated successfully!")

        logger.info(f"Converting start_date, end_date, and interval to datetime objects")
        start_date_dt = datetime_parser(start_date)
        end_date_dt = datetime_parser(end_date)
        if start_date_dt > end_date_dt:
            raise ValueError(f"Start date must be before end date. Found: {start_date_dt} > {end_date_dt}.")
        
        final_dataset_evalscripts = {}
        dataset_names = datasets_evalscripts_loaded.keys()

        for dataset_name in dataset_names:
            if dataset_name not in SUPPORTED_DATASET_EVALSCRIPTS.keys():
                logger.error(
                    f"Dataset {dataset_name} not supported. Use one of {SUPPORTED_DATASET_EVALSCRIPTS.keys()}"
                )
                sys.exit(1)
            requested_evalscripts = datasets_evalscripts[dataset_name]
            supported_evalscripts = [x['name'] for x in SUPPORTED_DATASET_EVALSCRIPTS[dataset_name]["supported_evalscripts"]]
            for evalscript in requested_evalscripts:
                if evalscript not in supported_evalscripts:
                    logger.error(
                        f"Evalscript {evalscript} not supported. Use one of {SUPPORTED_DATASET_EVALSCRIPTS[dataset_name]['supported_evalscripts']}"
                    )
                    raise ValueError(
                        f"Evalscript {evalscript} not supported for {dataset_name}. Use one of {SUPPORTED_DATASET_EVALSCRIPTS[dataset_name]['supported_evalscripts']}"
                    )
            final_dataset_evalscripts[dataset_name] = SUPPORTED_DATASET_EVALSCRIPTS[dataset_name]
            final_dataset_evalscripts[dataset_name]["evalscripts"] = [x for x in SUPPORTED_DATASET_EVALSCRIPTS[dataset_name]["supported_evalscripts"] if x["name"] in requested_evalscripts]
        
        logger.info(f"Setting up time travel for case study: {case_study_name}")


        kernel_planckster, protocol, file_repository = setup(
            job_id=job_id,
            logger=logger,
            kp_auth_token=kp_auth_token,
            kp_host=kp_host,
            kp_port=kp_port,
            kp_scheme=kp_scheme,
        )

        scraped_data_repository = ScrapedDataRepository(
            protocol=protocol,
            kernel_planckster=kernel_planckster,
            file_repository=file_repository,
        )

        root_relative_path = f"{case_study_name}/{tracer_id}/{job_id}"
        relevant_files = kernel_planckster.list_source_data(root_relative_path)

        if not relevant_files or len(relevant_files) == 0:
            logger.error(f"No relevant files found in {root_relative_path}.")
            sys.exit(1)
        print(relevant_files)

        logger.info(f"Scraper setup successfully for case study: {case_study_name}")
        logger.debug(f"__main__ function invoked with arguments: {locals()}")

    except Exception as e:
        logger.error(f"Unable to setup the swissgrid scraper. Error: {e}")
        sys.exit(1)

    logger.info(f"Scraping data for case study: {case_study_name}")

    scrape(
    case_study_name=case_study_name,
    job_id=job_id,
    tracer_id=tracer_id,
    scraped_data_repository=scraped_data_repository,
    log_level=log_level,
    latitude=latitude,
    longitude=longitude,  
    start_date=start_date_dt,
    end_date=end_date_dt,
    resolution=resolution,
    data_type=data_type,
    file_dir=file_dir,
    sentinel_client_id = sentinel_client_id,
    sentinel_client_secret = sentinel_client_secret,
    predict_url = predict_url,
    prediction_model_name = prediction_model_name
    )



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Scrape data from Sentinel datacollection.")

    parser.add_argument(
        "--case-study-name",
        type=str,
        default="webcam",
        help="The name of the case study",
    )

    parser.add_argument(
        "--job-id",
        type=int,
        default="1",
        help="The job id",
    )

    parser.add_argument(
        "--tracer-id",
        type=str,
        default="1",
        help="The tracer id",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        help="The log level to use when running the scraper. Possible values are DEBUG, INFO, WARNING, ERROR, CRITICAL. Set to WARNING by default.",
    )

    parser.add_argument(
        "--latitude",
        type=str,
        required=True,
        help="latitude of the location",
    )

    parser.add_argument(
        "--longitude",
        type=str,
        required=True,
        help="longitude of the location",
    )

    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Start datetime in the format 'YYYY-MM-DDTHH:MM",
    )

    parser.add_argument(
        "--end_date",
        type=str,
        required=True,
        help="End datetime in the format 'YYYY-MM-DDTHH:MM",
    )
    
    parser.add_argument(
        "--resolution",
        type=int,
        default=10,
        help="scraper resolution",
    )
    
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        help="data type for scraping",
    )
    
    parser.add_argument(
        "--sentinel_client_id",
        type=str,
        required=True,
        help="sentinel hub client id",
    )
    
    parser.add_argument(
        "--sentinel_client_secret",
        type=str,
        required=True,
        help="sentinel hub client secret",
    )

    parser.add_argument(
        "--predict_url",
        type=str,
        required=True, 
        help="URL for the prediction service",
    )

    parser.add_argument(
        "--prediction_model_name",
        type=str,
        required=True,
        help="Name of the prediction model to use",
    )

    parser.add_argument(
        "--datasets_evalscripts",
        type=str,
        required=True,
        help="Datasets and evalscripts to use for scraping",
    )
    
    parser.add_argument(
        "--kp_host",
        type=str,
        default="60",
        help="kp host",
    )

    parser.add_argument(
        "--kp_port",
        type=int,
        default="60",
        help="kp port",
    )

    parser.add_argument(
        "--kp_auth_token",
        type=str,
        default="60",
        help="kp auth token",
        )

    parser.add_argument(
        "--kp_scheme",
        type=str,
        default="http",
        help="kp scheme",
        )

    parser.add_argument(
        "--file_dir",
        type=str,
        default="./.tmp",
        help="saved file directory",
    )


    args = parser.parse_args()

    main(
        case_study_name=args.case_study_name,
        job_id=args.job_id,
        tracer_id=args.tracer_id,
        log_level=args.log_level,
        latitude=args.latitude,
        longitude=args.longitude,
        start_date=args.start_date,
        end_date=args.end_date,
        resolution=args.resolution,
        data_type=args.data_type,
        kp_host=args.kp_host,
        kp_port=args.kp_port,
        kp_auth_token=args.kp_auth_token,
        kp_scheme=args.kp_scheme,
        file_dir=args.file_dir,
        sentinel_client_id=args.sentinel_client_id,
        sentinel_client_secret=args.sentinel_client_secret,
        predict_url=args.predict_url,
        prediction_model_name=args.prediction_model_name,
        datasets_evalscripts=args.datasets_evalscripts
    )