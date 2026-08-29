"""
Project : InsightCart
File    : logger.py
Purpose : Configure project logging
"""

import logging
import os
from datetime import datetime

LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="[ %(asctime)s ] %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("InsightCart")
