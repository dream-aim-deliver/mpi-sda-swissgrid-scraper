import logging
import sys
from app.sdk.scraped_data_repository import ScrapedDataRepository
from app.setup import setup, string_validator
from app.time_travel.swissgrid_metadata_generator import generate_time_travel_metadata

def main(
    job_id: int,
    tracer_id: str,
    predict_url: str,
    prediction_model_name: str,
    kp_host: str,
    kp_port: int,
    kp_auth_token: str,
    kp_scheme: str,
    log_level: str = "WARNING"
) -> None:

    try:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

        case_study_name = "swissgrid"

        required_variables = [
            job_id,
            tracer_id,
            predict_url,
            prediction_model_name
        ] 
    
        for required_variable in required_variables:
            if not required_variable:
                raise ValueError(f"Required variable {required_variable} is missing.")

        string_variables = {
            "case_study_name": case_study_name,
            "job_id": job_id,
            "tracer_id": tracer_id,
        }

        logger.info(f"Validating string variables:  {string_variables}")

        for name, value in string_variables.items():
            string_validator(f"{value}", name)

        logger.info(f"String variables validated successfully!")

        logger.info(f"Converting start_date, end_date to datetime objects")

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

        job_output = generate_time_travel_metadata(
            job_id=job_id,
            tracer_id=tracer_id,
            scraped_data_repository=scraped_data_repository,
            relevant_source_data=relevant_files,
            protocol=protocol,
            predict_url=predict_url,
            prediction_model_name=prediction_model_name,
        )

        logger.info(f"{job_id}: Job finished with state: {job_output.job_state.value}")

        if job_output.job_state.value == "failed":
            sys.exit(1)

    except Exception as e:
        logger.error(f"Unable to setup the swissgrid scraper. Error: {e}")
        sys.exit(1)

    logger.info(f"Scraping data for case study: {case_study_name}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Scrape data from Sentinel datacollection.")

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


    args = parser.parse_args()

    main(
        job_id=args.job_id,
        tracer_id=args.tracer_id,
        log_level=args.log_level,
        kp_host=args.kp_host,
        kp_port=args.kp_port,
        kp_auth_token=args.kp_auth_token,
        kp_scheme=args.kp_scheme,
        predict_url=args.predict_url,
        prediction_model_name=args.prediction_model_name,
    )