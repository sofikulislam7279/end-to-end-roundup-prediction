from roundup.logger import logging
from roundup.exception import RoundupException
import sys

try:
    r = 1 / 0
    print(r)

except Exception as e:
    logging.error("An error occurred: %s", e)

else:
    logging.info("Code executed successfully.")

finally:
    logging.info("Execution completed.")

