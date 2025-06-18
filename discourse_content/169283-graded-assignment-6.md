@Jivraj Singh Shekhawat said: Please post any questions related to [Graded Assignment 6 \- Data Analysis](https://seek.onlinedegree.iitm.ac.in/courses/ns_25t1_se2002?id=25&type=assignment&tab=courses&unitId=23)


Please use markdown code formatting (fenced code blocks) when sharing code (rather than screenshots). It’s easier for us to copy\-paste and test.


Deadline 2025\-03\-15T18:30:00Z


@Jivraj Singh Shekhawat said: 
@Lovepreet Singh said: The answer choices for questions 1 and 2 in graded assignment 6 are quite confusing. Both questions are single\-select, yet three out of the four options are correct in each case. I’m unsure whether to choose one of the correct options or if the question is actually asking for the incorrect one. Could someone please clarify?


[@carlton](/u/carlton)


@Sarang Tambe said: [@Jivraj](/u/jivraj) [@Saransh\_Saini](/u/saransh_saini)  

I have similar concern  

For Q1, I used the following code:



```
print(f'Pearson correlation for Karnataka between price retention and column')
kk = df[df['State'] == 'Karnataka']
for col in ['Mileage (km/l)', 'Avg Daily Distance (km)', 'Engine Capacity (cc)']:
    pearson_corr = kk['price_retention'].corr(kk[col])
    print(f'\t{col:25} : {pearson_corr:.2f}')

```

And got the following output:



```
Pearson correlation for Karnataka between price retention and column
	Mileage (km/l)            : 0.03
	Avg Daily Distance (km)   : -0.06
	Engine Capacity (cc)      : -0.04

```

Whereas options are below where none of them are correct.  

[![image](https://europe1.discourse-cdn.com/flex013/uploads/iitm/original/3X/a/6/a6fa9a2e601c94da84cbd25c406235d1009b204c.png)image281×219 9\.1 KB](https://europe1.discourse-cdn.com/flex013/uploads/iitm/original/3X/a/6/a6fa9a2e601c94da84cbd25c406235d1009b204c.png "image")


Whereas for Q2 (Punjab and Yamaha) I used the following code:



```
print(f'Pearson correlation for Punjab and Yamaha between price retention and column')
pb = df[(df['State'] == 'Punjab') & (df['Brand'] == 'Yamaha')]
for col in ['Mileage (km/l)', 'Avg Daily Distance (km)', 'Engine Capacity (cc)']:
    pearson_corr = pb['price_retention'].corr(pb[col])
    print(f'\t{col:25} : {pearson_corr:.2f}')

```

and got the following answers:



```
Pearson correlation for Punjab and Yamaha between price retention and column
	Mileage (km/l)            : 0.24
	Avg Daily Distance (km)   : -0.06
	Engine Capacity (cc)      : -0.08

```

The options for Q2 are given below and 2 of them are correct (AvgDistance and Mileage).  

[![image](https://europe1.discourse-cdn.com/flex013/uploads/iitm/original/3X/5/1/51b03d00c3e962e6c4fc7fc64930a23e82500006.png)image278×216 9\.19 KB](https://europe1.discourse-cdn.com/flex013/uploads/iitm/original/3X/5/1/51b03d00c3e962e6c4fc7fc64930a23e82500006.png "image")


@Carlton D'Silva said: [@24f2006061](/u/24f2006061) We are looking into it. We will update based on our analysis. Thanks for letting us know.


Kind regards


@Abhinav said: I used a python script to get the solution to quesiton 1 of week 6 graded assignment. It matches three options. Is this a bug or like we then need to analyze using the pearson coefficient to determine which option is the correct one  

[![image](https://europe1.discourse-cdn.com/flex013/uploads/iitm/optimized/3X/b/d/bd0ea5ffab782a7d6bcc8b1cde7ba7f385b85630_2_690x131.png)image1383×263 25 KB](https://europe1.discourse-cdn.com/flex013/uploads/iitm/original/3X/b/d/bd0ea5ffab782a7d6bcc8b1cde7ba7f385b85630.png "image")


@Wasim Ansari said: Dear Sirs, Can we have some response on these issues related particularly to the questions 1 and 2 of Graded Assignment 6\. It looks like multiple options are correct in the given options. Any guidance or hint, on how to arrive at the right answer will be helpful. Thanks and regards. [@carlton](/u/carlton) [@Jivraj](/u/jivraj) [@Saransh\_Saini](/u/saransh_saini)


@Shashannk said: Yeah…Even I am facing the same issue. Out of the 4 options provided, 3 options are correct in my case both for Q1 \& Q2, but both these questions are single\-choice questions. Kindly look into it and help us out [@carlton](/u/carlton) !


@RAJ K BOOPATHI said: I guess for both Q1 \& Q2, we need to find the option that is having stronger correlation (positive/negative). Please correct me if I am wrong.


@Pradeep Mondal said: Any updates on these? I am too facing the same issue.


[@carlton](/u/carlton) [@Jivraj](/u/jivraj) [@Saransh\_Saini](/u/saransh_saini)


@Udipth said: In GA6 for first 2 questions 3 out of 4 options are correct. Even the question is not clearly asking anything. Kindly suggest are we supposed to select the wrong one  

[![image](https://europe1.discourse-cdn.com/flex013/uploads/iitm/optimized/3X/f/c/fccc54e8cff0595d93b1c5185ce0a10343849b04_2_690x190.png)image2083×575 47\.6 KB](https://europe1.discourse-cdn.com/flex013/uploads/iitm/original/3X/f/c/fccc54e8cff0595d93b1c5185ce0a10343849b04.png "image")


@Shashannk said: Kindly update us regarding the status of Q1 \& Q2 [@carlton](/u/carlton) [@Jivraj](/u/jivraj)


@LAKSHAY said: [@Jivraj](/u/jivraj) [@carlton](/u/carlton) [@Saransh\_Saini](/u/saransh_saini)  

Dear TDS Team,


There are multiple issues in Graded Assignment 6 that require urgent attention:


1. Questions 1 and 2, along with their options, are ambiguous.
2. In Questions 3 and 4, I am unable to obtain an exact answer that matches any of the given options, despite trying multiple approaches, including the Excel regression method and other models in a Google Colab file.
3. The data for Question 10 is missing. I attempted to run the shapefile in QGIS, but it resulted in an error. Additionally, I searched for the shapefile of New York roads on official websites, but their servers are currently under maintenance.


The assignment deadline is approaching, but these issues remain unresolved. Kindly look into this matter at the earliest and provide a resolution as soon as possible.


Thank you for your support.


@Pradeep Mondal said: Yes, there are no specifics in Q1 to Q4 and are quite ambiguous.


For instance:



> forecast the 2027 resale value of the Hero \- HF Deluxe in Gujarat, using historical data.


but is this talking about the average resale value as no input features are specified?


@LAKSHAY said: Let’s wait for their response.  

I submitted nearby option for Q3 and Q4


@Vivek Rekha Ashoka said: [@Jivraj](/u/jivraj) [@carlton](/u/carlton) [@Saransh\_Saini](/u/saransh_saini)  

Can you please provide any update ASAP as the deadline for this GA coincides with Quiz 2\. With many ambiguities unresolved it’s hard to solve this and study for Quiz 2 (and do offline college work even though that’s not your problem).  

Thanks


@Jivraj Singh Shekhawat said: Hi @all


Question intends you to select most correlated one.  

Select option which is absolute highest.


@M v Sunil said: [@Jivraj](/u/jivraj) \- Can you please check answer choices for Q7 for GA6 where no choices are matching with the answer. The answer is coming to around 11\.5 kms which is 11500 meters.  

Q.A wildfire is threatening a rural mountain region, and emergency services need to coordinate evacuation routes for four remote communities. The Emergency Management Center is located at a central command post, and must plan the most efficient evacuation route to ensure rapid and safe community evacuation. The four communities are: Pine Pines Junction : (26\.5596,\-99\.5336\) ;Maple Fields Station : (26\.4212,\-99\.4597\);South Glen Crossing : (26\.5962,\-99\.5243\);Cedar Creek Retreat : (26\.56,\-99\.4519\) \& Central Command Post Location: (26\.4644,\-99\.4771\) Using the Haversine package, calculate the distance from the Central Command Post to Pine Pines Junction. Which of the following is the MOST ACCURATE distance


@Shashank Tripathi said: [![image](https://europe1.discourse-cdn.com/flex013/uploads/iitm/optimized/3X/9/6/9656b143021a1b4baf78510b1ba05ae9cbd6ca9b_2_690x197.png)image1318×377 34\.2 KB](https://europe1.discourse-cdn.com/flex013/uploads/iitm/original/3X/9/6/9656b143021a1b4baf78510b1ba05ae9cbd6ca9b.png "image")  

what to do if 3 options have same value \-0\.04 and all are correct?


@Khushi Shah said: [@carlton](/u/carlton) [@Jivraj](/u/jivraj)  

My question 7 for GA6 is :  

A wildfire is threatening a rural mountain region, and emergency services need to coordinate evacuation routes for four remote communities. The Emergency Management Center is located at a central command post, and must plan the most efficient evacuation route to ensure rapid and safe community evacuation. The four communities are: Silver Springs Community : (42\.1029,\-85\.665\) ;Pleasant Harbor Community : (42\.1238,\-85\.9043\);Summit Shores Village : (42\.0415,\-85\.8696\);River Retreat Outpost : (42\.0417,\-85\.6836\) \& Central Command Post Location: (42\.0587,\-85\.7226\) Using the Haversine package, calculate the distance from the Central Command Post to Silver Springs Community. Which of the following is the MOST ACCURATE distance  

Whose options provided are :  

10418 meters


12287 meters


10965 meters


11149 meters


However, after trying all methods out there my distance comes out to be 6873 meters, I selected 10418 as the answer (closest approximation to 6873 meters)


I assume that the question must have been central command post to summit shores village (whose answer turns out to be 12287 meters)  

Kindly look into the question, and let me know about the same (the destination from central command post)

