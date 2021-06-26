#_*_ coding:utf-8_*_
import pandas as pd
import matplotlib.pyplot as plt

women_degrees = pd.read_csv('./DATA/regress/percent-bachelors-degrees-women-usa.csv')
print(women_degrees.shape)
stem_cats = ['Agriculture','Architecture','Art and Performance','Biology','Business','Communications and Journalism','Computer Science','Education','Engineering','English','Foreign Languages','Health Professions','Math and Statistics','Physical Sciences','Psychology','Public Administration','Social Sciences and History']

cb_dark_blue = (0/255, 107/255, 164/255)
cb_orange = (255/255, 128/255, 14/255)

fig = plt.figure(figsize=(20, 10))

for sp in range(0, 17):
    ax = fig.add_subplot(3, 6, sp + 1)
    ax.plot(women_degrees['Year'], women_degrees[stem_cats[sp]], c=cb_dark_blue, label='Women', linewidth=3)
    ax.plot(women_degrees['Year'], 100 - women_degrees[stem_cats[sp]], c=cb_orange, label='Men', linewidth=3)
    for key, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xlim(1968, 2011)
    ax.set_ylim(0, 100)
    ax.set_title(stem_cats[sp])
    ax.tick_params(bottom=False, top=False, left=False, right=False)

    if sp == 0 or 6 or 13:
        ax.text(2005, 87, 'Men')
        ax.text(2002, 8, 'Women')
    elif sp == 5 or 11 or 16:
        ax.text(2005, 62, 'Men')
        ax.text(2001, 35, 'Women')
plt.legend(loc='best')
plt.show()