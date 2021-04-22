import numpy  as np
import pandas as pd

#Preprocessing
from sklearn import preprocessing
import nltk
import pandas as pd
import time
import re
from nltk.stem import PorterStemmer

def cleanData(df):
	start = time.time()

	df.drop('S. No.',axis='columns', inplace=True)
	df.drop('Recommended by (HOD) (YES/NO)',axis='columns', inplace=True)
	df.drop('Approved by (Director Sir) (YES/NO)',axis='columns', inplace=True)
	df.drop('Dates',axis='columns', inplace=True)
	df.drop('No. of Days',axis='columns', inplace=True)
	df.replace('', np.nan)
	l=df.filter(["Name of Faculty/Staff", "Department","Organised by University/Institution/Organisation"]).mode()
	df[["Name of Faculty/Staff", "Department","Organised by University/Institution/Organisation"]]=df[["Name of Faculty/Staff", "Department","Organised by University/Institution/Organisation"]].fillna(value=l.iloc[0])
	df["FDP/Conferences Details"].fillna( df["FDP/Conferences Details"].mode()[0] , inplace=True )
	df.rename(columns = {'Organised by University/Institution/Organisation': 'University', 'Name of Faculty/Staff': 'Faculty', 'FDP/Conferences Details': 'Conference_Details'}, inplace = True)
	df.Department.replace(['CS&IT','CS/IT','CS and IT','CS & IT','IT','CS-IT','CS &IT','CS','Computer'],'CS and IT' , inplace=True)
	df.Department.replace(['Civil','Civil '],'Civil' , inplace=True)
	df.Department.replace(['Applied Science','Appled Science','Applied Sc','Applied Sc.','Applied Sciences','Applied. Science','App Science'],'Applied Science' , inplace=True)
	df.Department.replace(['E&TC','ENTC','E & TC','E &TC'],'ENTC' , inplace=True)
	df.Department.replace(['Mechanical','Mech Engg','Mechnical','Mechanical Eng','Mechanical ','Mech'],'Mechanical' , inplace=True)
	df.University.replace(["Conference Room, ELTIS, Model Colony, Pune",'Conference room','ELTIS Model Colony','ELTIS, Model Colony, Pune','ELTIS,Model colony','ELTIS'],'ELTIS Model Colony' , inplace=True)
	df.University.replace(['STLTC','STLRC','TLRC','STLRC SIU',
	                       'STLRC, SIU at SIT',
	                       'STLRC, SIU ( Venue : SIT )',
	                       'TLRC, SIU','STLRC, SIU','TLRC'
	                       'TLRC,SIU (Venue:Conference Room, ELTIS, Model',
	                       'STLRC at conference hall SIU','STLRC at SID viman nagar', 
	                       'STLRC at Convention Hall SIU',
	                       'STLRC at conference hall SIU and SIMS Kharki',
	                       'STLRC at conventional hall SIU ', 'STLRC at CAD CAM LAB SIT',
	                       'STLRC,SIU,Pune','STLRC, SIU, Pune','TLRC ,','TLRC , SIU','yes'],'STLRC, SIU' , inplace=True)
	df.University.replace(['SIT and STLRC ( Venue : SIT )','SIT & STLRC, SIU',
	                       'SIT, STLRC','STLRC at SIT','STLRC, at SIT','SIT & STLRC'
	                       'SIT &STLRC','SIT Civil, STLRC','SIT in collaboration with STLRC',
	                       'SIT & STLRC','SIT &STLRC','SIT\xa0&STLRC '],'STLRC and SIT' , inplace=True)
	df.University.replace(['SIMS and STLRC, Pune','SIMS and STLRC, SIU','SIMS (Khirkee), STLRC',
	                        'STLRC, Computer Lab, SIMS Kirkee','STLRC at SIMS Kirkee'],'SIMS and STLRC' , inplace=True)
	df.University.replace(['SIBM and STLRC, SIU','SIBM, Lavale and STLRC','STLRC, SIBM, SIU','SIBM in Collaboration with STLRC',
	                       'STLRC SIBM SIU'],'SIBM and STLRC' , inplace=True)
	df.University.replace(['SIIB, Hinjewadi','SIIB, SIU, Pune',],'SIIB Hinjewadi' , inplace=True)
	df.University.replace(['TLRC, SIU ( Venue : SCMHRD, Hinjewadi )','TLRC,SCMHRD Pune','TLRC, SCMHRD','SCMHRD, Hinjewadi Pune'],'STLRC at SCMHRD Hinjewadi' , inplace=True)
	df.University.replace(['SCMS and STLRC, SIU','SCMS, Viman Nagar Pune'],'SCMC Viman Nagar' , inplace=True)
	df.University.replace(['Dept. of Civil Engineering, NITTTR Chennai and STLRC','SIT, STLRC and NITTTR, Chennai','STLRC & SIT, Pune,NITTTR Chennai','SIT in Collaboration with STLRC and National Institute of Technical Teachers Training Institute, HRD Ministry, Govt. of India.','SIT, STLRC and NIITR Chennai','SIT, STLRC,NITTTR','STLRC + NITTTIR + SIT civil Dept, Pune'],'SIT, STLRC and NIITR Chennai' , inplace=True)
	df.University.replace(['SIU', 'SIU, Lavale Campus','SIU, Pune','SIU,Pune', 'SIU, at SLS Viman nagar', 'SIU,Pune\n', 'Convention Hall, SIU, Pune','Convention Hall, SIU, Lavale\n', 'MDP Hall SIBM, SIU Lavale','siu','Faculty of Humanities & Social Science,SIU' ],'SIU Lavale Campus' , inplace=True)
	df.University.replace(['SIT','SIT, Pune','SIT,Pune','sit','SIT,Lavale','SIT, SIU', 'SIT civil dept.'],'Symbiosis Institute of Technology' , inplace=True)
	df.University.replace(['IIT, Mumbai','IIT Delhi, New Delhi','IIT, Roorkee','IIT Roorkee, Uttarakhand','IIT Gandhinagar','IIT Bombay','IIT Roorkee', 'IIT Indore'],'Indian Institute of Technology' , inplace=True)
	df.University.replace(['SIBM','SIBM, SIU'],'Symbiosis Institute of Buisness Management' , inplace=True)
	df.University.replace(['Department of Computer Engineering & Information TechnologyCollege of Engineering, Pune','COEP, Pune','Department of I & C, COEP,Pune','COEP and AICTE','COEP'],'College of Engineering Pune' , inplace=True)
	time.sleep(1)

	end = time.time()
	print(f"Runtime of the program is {end - start}")


	return df


def encodeData(df):
	le = preprocessing.LabelEncoder()
	df = df.apply(le.fit_transform)
	return df


