import csv
import time
start = time.time()
with open('../diabetes_data_upload.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

sum_age = 0
sum_age_for_male, sum_age_for_female = 0, 0
male_count, female_count = 0, 0
for row in data[1:]:
    sum_age += int(row[0])
    if row[1] == "Male":
        sum_age_for_male += int(row[0])
        male_count += 1
    else:
        sum_age_for_female += int(row[0])
        female_count += 1
avg_age = sum_age / len(data[1:])
avg_age_for_female = sum_age_for_female / female_count
avg_age_for_male = sum_age_for_male / male_count

print(f"Avarage age is: {avg_age}")
print(f"Avarage age for female is: {avg_age_for_female}")
print(f"Avarage age for male is: {avg_age_for_male}")
print(f"Execution time: {time.time() - start}")


print("-" * 15)

both = []
diabetes = []
obesity = []
none = []
for i in range(1, len(data)):
    if data[i][-1] == "Positive" and data[i][-2] == "Yes":
        both.append(i)
    elif data[i][-1] == "Positive" and data[i][-2] == "No":
        diabetes.append(i)
    elif data[i][-1] == "Negative" and data[i][-2] == "Yes":
        obesity.append(i)
    else:
        none.append(i)

table = [
    ["", "Diabetes", ""],
    ["Obesity", "Positive (1)", "Negative (0)"],
    ["Positive (1)", len(both), len(obesity)],
    ["Negative (0)", len(diabetes), len(none)]
]

col_widths = [max(len(str(row[i])) for row in table) for i in range(len(table[0]))]

for row in table:
    print("| " + " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row))) + " |")
    if row == table[1]:
        print("|-" + "-|-".join("-" * col_widths[i] for i in range(len(row))) + "-|")

print(f"Both: {both}")
print(f"Diabetes: {diabetes}")
print(f"Obesity: {obesity}")
print(f"None: {none}")
