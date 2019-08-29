import matplotlib 
matplotlib.use('agg')
import seaborn as sns
import sys 
sys.path.append('src')

from data import read_processed_data

def exploratory_visualization(df):
    return sns.pairplot(df)


def main(input_file, output_file):
    print('Plotting pairwise distribution...')

    regression_variables = ['temp','month','weekday','days','windspeed', 
                            'cloud_cover', 'indoorTemp']
    
    df = read_processed_data(input_file)
    plot = exploratory_visualization(df)
    plot.savefig(output_file)

if __name__ == '__main__':
    main()