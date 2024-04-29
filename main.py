import matplotlib.pyplot as plt
import pandas as pd
import os


def create_file_if_not_exists(file_path):

    if not os.path.exists(file_path):  # Check if the file doesn't exist
        # If the file doesn't exist, create it
        with open(file_path, 'w') as file:
            file.writelines('0, 0')
            return 0, 0
    else:
        # If the file exists, clear it
        with open(file_path, 'r') as file:
            line = file.readline()
            theta0, theta1 = line.split(',')
            return theta0, theta1

def check_input(km):
    if km < 0:
        print('\033[91m'+'Mileage must be a positive number'+'\033[0m')
        return False
    if not km:
        print('\033[91m'+'Mileage must be a number'+'\033[0m')
        return False
    if km > 240000:
        print('\033[91m'+'Mileage must be less than 240000'+'\033[0m')
        return False
    return True

def main():
    # Read the data
    try:
        data_file = input('\033[92m'+'Data file: '+'\033[0m')
        data = pd.read_csv(data_file)
        theta0, theta1 = create_file_if_not_exists('model.csv')
        km = int(input('\033[93m'+'Mileage: '+'\033[0m'))
        if not check_input(km):
            return
        print('\033[93mPrice: ', float(theta0) + float(theta1) * km, '\033[0m')
        plt.title('ft_linear_regression')
        plt.xlabel('Km')
        plt.ylabel('Price')
        plt.scatter(data['km'], data['price'])
        plt.plot(data['km'], [float(theta0) + float(theta1)
                              * km for km in data['km']], color='red')
        plt.show()
    except Exception as e:
        print('Exception:', '\033[91m', e, '\033[0m')


if __name__ == '__main__':
    main()
