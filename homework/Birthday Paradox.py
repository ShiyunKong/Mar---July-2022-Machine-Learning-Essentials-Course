'''
Suppose that we have a class of n students. We assume that their birthdays are uniformly distributed over the days of the year.
Question I: What is the probability that the class has two students having the same birthday? Write python code to simulate this.
Question II: For what value of n (number of students in the class) the above probability is 0.5? Use the Python code you developed in question I, to find that value by trial and error (or a more sophisticated algorithm)
'''

students = 2
probability = 365 / 365
divisor = 365
for i in range(1, students+1):
	multiplier = (365 - i + 1)
	probability *= multiplier
	probability /= divisor
final_probability = 1 - probability
print(f'The probability of having {students} students share the same birthday is :\n{final_probability*100}%')
while probability >= 0.5:
	students += 1
	multiplier = (365 - students + 1)
	probability *= multiplier
	probability /= divisor
fifty_percent_benchmark = 1 - probability
print(f"When there is {students} students, there shall be {fifty_percent_benchmark*100}% chance of having same birthday(s) amongst them.")
