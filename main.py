from api import create_app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=21200, debug=True, auto_reload=True)