# Smartfolio: Optimal Portfolio Advisor

## Overview
Smartfolio is a Flask-based web application designed to assist users in determining their optimal investment portfolio based on individual risk profiles and investment horizons. The app utilizes historical financial data and machine learning predictions to recommend asset allocation strategies.

## Features
- **Dynamic Risk Assessment:** Users answer a series of questions to determine their risk tolerance and investment timeline.
- **Machine Learning Integration:** Utilizes machine learning models to predict future asset prices and optimize portfolio allocations accordingly.
- **Responsive Design:** Built with Bootstrap for a responsive design that works on desktops and mobile devices.

## Technology Stack
- **Flask:** Serves the web application and handles backend logic.
- **Pandas & NumPy:** For data manipulation and calculations.
- **Yahoo Finance API:** Fetches historical asset prices.
- **Sklearn:** Machine learning algorithms for predicting asset prices.
- **Bootstrap:** Frontend framework for responsive design.

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/Tom-Scheer/Fintech_mvp.git
    cd Fintech-mvp
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application
1. Start the Flask application:
    ```bash
    python app.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000` to start using the application.

## Usage
- **Home Page:** Start by answering the questionnaire to assess your risk profile.
- **Results Page:** View your recommended portfolio, with details about each asset's allocation.

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests to the `develop` branch.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Project link
- Project Link: [https://github.com/Tom-Scheer/Fintech_mvp](https://github.com/Tom-Scheer/Fintech_mvp)

## Acknowledgements
- [Yahoo Finance](https://finance.yahoo.com) for providing financial data.
- [Bootstrap](https://getbootstrap.com) for the responsive frontend framework.
- [Flask](https://flask.palletsprojects.com/) for backend framework support.
