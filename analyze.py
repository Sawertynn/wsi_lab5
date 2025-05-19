import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron_2_layers, laplace, MSE, MAE

MIN_X = -8
MAX_X = 8
SAMPLE_COUNT = 100
TRIES = 5
EPOCHS = 50_000
LEARNING_RATE = 0.01
SIZES = [4, 7, 10, 13, 20]

SHOW_PLOT = False
TEST_UNNORMAL = False

def run_show(x, y, normalize=True):
    mses = {}
    maes = {}
    for hidden_size in SIZES:
        mses[hidden_size] = []
        maes[hidden_size] = []
        print(f'== SIZE {hidden_size} ==')
        y_pred_best = None
        best_mse = np.inf
        for i in range(TRIES):
            model = Perceptron_2_layers(hidden_size)
            if normalize:
                model.train(x, y, EPOCHS, LEARNING_RATE, printing=False)
                y_pred = model.predict(x)
            else:
                model.train_unnormalized(x, y, EPOCHS, LEARNING_RATE, printing=False)
                y_pred = model.predict_unnormalized(x)
            mse = MSE(y, y_pred)
            mae = MAE(y, y_pred)

            if y_pred_best is None or mae < best_mse:
                y_pred_best = y_pred
                best_mse = mae
            mses[hidden_size].append(float(mse))
            maes[hidden_size].append(float(mae))
            

        # plot the results
        mode = 'normal' if normalize else 'unnorm'
        plt.clf()
        plt.plot(x, y, label='True')
        plt.plot(x, y_pred_best, label='Predicted')
        plt.title(f'{mode}, hidden size: {hidden_size}, MSE: {mse:.2e}, MAE: {mae:.2e}')
        plt.legend()
        if SHOW_PLOT:
            plt.show()
        else:
            path = f'plots_8/{mode}_size={hidden_size}'
            plt.savefig(path)
    
    
    print("mean square error")
    print("| size | min | max |")
    print("| -- | ------- | -------- |")
    for size in SIZES:
        vals = mses[size]
        print(f'| {size:2} | {min(vals):.2e} | {max(vals):.2e} |')

    
    
    print("mean absolute error")
    print("| size | min | max |")
    print("| -- | ------- | -------- |")
    
    for size in SIZES:
        vals = maes[size]
        # print(f'size: {size:2}, MAE: min {min(vals):.2e} max: {max(vals):.2e}')
        print(f'| {size:2} | {min(vals):.2e} | {max(vals):.2e} |')

def main():
    x = np.linspace(MIN_X, MAX_X, SAMPLE_COUNT).reshape(-1, 1)
    y = laplace(x).reshape(-1, 1)

    print(' ## NORMALIZED ## ')
    run_show(x, y, True)
    if TEST_UNNORMAL:
        print(' ## NOT Normalized ## ')
        run_show(x, y, False)

if __name__ == '__main__':
    main()