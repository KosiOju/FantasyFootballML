from api.app import create_app


application = create_app()

if __name__ == '__main__':
	application.run()
	# 1st
	# pip install -r packages\ml_api\requirements.txt
	# 2nd
	# set FLASK_APP=run.py
	# 3rd
	# python run.py