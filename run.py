from app import server
import logging

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(filename = 'app.log', format=FORMAT, level =logging.INFO)

if __name__ == "__main__":
    server.main()
