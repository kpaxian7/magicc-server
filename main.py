from api import create_app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=21200, auto_reload=False, workers=2)
    # app.run(host='0.0.0.0', port=21200, auto_reload=False, workers=2, debug=True)
