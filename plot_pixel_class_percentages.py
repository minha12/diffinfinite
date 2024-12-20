
import csv
import matplotlib.pyplot as plt

# Read the CSV file
data = []
with open('pixel_class_percentages.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(row)

# Sort the data by percentage
data.sort(key=lambda x: float(x['Percentage']), reverse=True)

# Extract class names and percentages
class_names = [row['Class Name'] for row in data]
percentages = [float(row['Percentage']) for row in data]

# Plot the bar graph
plt.figure(figsize=(15, 10))
plt.barh(class_names, percentages, color='skyblue')
plt.xlabel('Percentage of Pixels')
plt.ylabel('Class Name')
plt.title('Percentage of Pixels per Class')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest percentage on top
plt.tight_layout()
plt.savefig('pixel_class_percentages.png', bbox_inches='tight', dpi=300)
plt.show()