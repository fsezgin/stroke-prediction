import matplotlib.pyplot as plt
import seaborn as sns

# I created the graph class to show relation between features and 'stroke'
class StrokeDataVisualizer:
    def __init__(self, data):
        self.data = data

    # with the help of this function I saw the 'others' part and deleted the line
    def count_gender_graph(self):
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(6, 5))
        sns.countplot(x= self.data['gender'], hue= self.data['stroke'],data= self.data, palette='twilight')
        plt.show()

    def count_hypertension_graph(self):
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(6, 5))
        sns.countplot(x=self.data['hypertension'], hue=self.data['stroke'], data=self.data, palette='twilight')
        plt.show()

    def count_age_graph(self):
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(20, 20))
        sns.countplot(x= self.data['age_group'], hue= self.data['stroke'],data= self.data, palette='twilight')
        plt.show()

    def count_heart_disease_graph(self):
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(6, 5))
        sns.countplot(x=self.data['heart_disease'], hue=self.data['stroke'], data=self.data, palette='twilight')
        plt.show()

    def count_ever_married_graph(self):
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(6, 5))
        sns.countplot(x=self.data['ever_married'], hue=self.data['stroke'], data=self.data, palette='twilight')
        plt.show()

    def count_work_type_graph(self):
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(6, 5))
        sns.countplot(x=self.data['work_type'], hue=self.data['stroke'], data=self.data, palette='twilight')
        plt.show()

    def count_residence_type_graph(self):
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(6, 5))
        sns.countplot(x=self.data['Residence_type'], hue=self.data['stroke'], data=self.data, palette='twilight')
        plt.show()

    def count_avg_glucose_lvl_graph(self):
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(6, 5))
        sns.countplot(x=self.data['avg_glucose_level'], hue=self.data['stroke'], data=self.data, palette='twilight')
        plt.show()

    def count_bmi_graph(self):
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(6, 5))
        sns.countplot(x=self.data['bmi_group'], hue=self.data['stroke'], data=self.data, palette='twilight')
        plt.show()

    # with the help of this function I saw the 'unkown' part and deleted the line
    def count_smoking_status_graph(self):
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(6, 5))
        sns.countplot(x=self.data['smoking_status'], hue=self.data['stroke'], data=self.data, palette='twilight')
        plt.show()

#*************************
    def dis_age_graph(self):
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(6, 5))
        sns.displot(data=self.data, x=self.data['age'], kde=True)
        plt.show()
    def dis_bmi_graph(self):
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(6, 5))
        sns.displot(data=self.data, x=self.data['bmi'], kde=True)
        plt.show()
    def dis_hypertension_graph(self):
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(6, 5))
        sns.displot(data=self.data, x=self.data['hypertension'], kde=True)
        plt.show()
    def dis_heart_disease_graph(self):
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(6, 5))
        sns.displot(data=self.data, x=self.data['heart_disease'], kde=True)
        plt.show()
    def dis_avg_glucose_level_graph(self):
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(6, 5))
        sns.displot(data=self.data, x=self.data['avg_glucose_level'], kde=True)
        plt.show()