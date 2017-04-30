from app import server
import logging

# create logger for app
logger = logging.getLogger('app')
logger.setLevel(logging.INFO)

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename = 'applog.log', format=FORMAT)

if __name__ == "__main__":
    server.main()