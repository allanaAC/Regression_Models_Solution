import logging
import warnings
from src.data.load_dataset import load_and_preprocess_data
from src.feature.build_features import create_features
from src.model.train_model import split_data, train_linear_reg, train_decision_tree, train_random_forest, save_model, load_model
from src.model.predict_model import make_predictions, evaluate_model
from src.visualization.visualize import plot_decision_tree

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    # Load and preprocess the data
    data_path = "src/data/final.csv"
    df = load_and_preprocess_data(data_path)

    # Prepare data
    x, y = create_features(df)

    # Split data
    x_train, x_test, y_train, y_test = split_data(df, x, y)

    # Train model
    model = train_linear_reg(x_train, y_train)

    # Make predictions
    #train_pred = make_predictions(model, x_train)
    #test_pred = make_predictions(model, x_test)
    
    # Train and evaluate Linear Regression model
    train_mae, test_mae = evaluate_model(model, x_train, x_test, y_train, y_test)
    
    print('Linear Regression - Train MAE: ', train_mae)
    print('Linear Regression - Test MAE: ', test_mae)

    # Train and evaluate Decision Tree model
    dt_model = train_decision_tree(x_train, y_train)
    dt_train_mae, dt_test_mae = evaluate_model(dt_model, x_train, x_test, y_train, y_test)
    print('Decision Tree - Train MAE: ', dt_train_mae)
    print('Decision Tree - Test MAE: ', dt_test_mae)

    # Plot Decision Tree
    plot_decision_tree(dt_model, x.columns)
    
     # Train and evaluate Random Forest model
    rf_model = train_random_forest(x_train, y_train)
    rf_train_mae, rf_test_mae = evaluate_model(rf_model, x_train, x_test, y_train, y_test)
    print('Random Forest - Train MAE: ', rf_train_mae)
    print('Random Forest - Test MAE: ', rf_test_mae)

    # Save the Random Forest model
    save_model(rf_model, 'RE_Model.pkl')

    # Load the saved model and make a prediction
    loaded_model = load_model('RE_Model.pkl')
    sample_prediction = loaded_model.predict([[2012, 216, 74, 1, 1, 618, 2000, 600, 1, 0, 0, 6, 0]])
    print('Sample prediction:', sample_prediction)

    