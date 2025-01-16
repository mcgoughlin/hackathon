import pandas as pd

# Load the data
dateOfDiagnosis = pd.read_csv('/Users/mcgoug01/Downloads/ARTIST imaging request_unidentifiable.csv')
studyMetadata = pd.read_csv('/Users/mcgoug01/Downloads/StudyListOut_unidentifiable.csv')

# Display the first few rows of the data
print(dateOfDiagnosis.head())
print(studyMetadata.head())
studyMetadata[' AnonPatientID'] = studyMetadata[' AnonPatientID'].astype(str).str.strip()
dateOfDiagnosis['Study ID'] = dateOfDiagnosis['Study ID'].astype(str).str.strip()

#append the date of diagnosis to the study metadata based on 'AnonPatientID' in studyMetadata and 'Study ID' in dateOfDiagnosis
studyMetadata['dateOfDiagnosis'] = studyMetadata[' AnonPatientID'].map(dateOfDiagnosis.set_index('Study ID')['Date of Diagnosis'])

# ' StudyDate' column currently is inappropriately formatted, convert it to a datetime object.
# currently, study date is saved as a number of YYYYMMDD, convert it to a datetime object
studyMetadata[' StudyDate'] = studyMetadata[' StudyDate'].astype(str)
studyMetadata[' StudyDate'] = studyMetadata[' StudyDate'].str.strip()
studyMetadata['StudyDate'] = pd.to_datetime(studyMetadata[' StudyDate'], format='%Y%m%d', errors='coerce')


#find the three study dates in studyMetadata (under column ' StudyDate') that is closest to the date of diagnosis per patient
studyMetadata['dateOfDiagnosis'] = pd.to_datetime(studyMetadata['dateOfDiagnosis'])
studyMetadata['dateDifference'] = studyMetadata['StudyDate'] - studyMetadata['dateOfDiagnosis']
studyMetadata_perpatient = studyMetadata.groupby(' AnonPatientID').apply(lambda x: x.nsmallest(3, 'dateDifference')).reset_index(drop=True)

#convert the dateDifference into hours, plot histogram of the dateDifference
studyMetadata_perpatient['dayDifference'] = (studyMetadata_perpatient['dateDifference'].dt.total_seconds())/(3600*24)


studyMetadata_perpatient[' StudyDescription'] = studyMetadata_perpatient[' StudyDescription'].str.strip().str.lower().str.replace(' ', '')
# set_a is the set of studyMetadata_perpatient that has 'contrast' in its description
set_a = studyMetadata_perpatient[studyMetadata_perpatient[' StudyDescription'].str.contains('contrast')]
# set_b is the set of studyMetadata_perpatient that has the shortest date difference to the date of diagnosis
set_b = studyMetadata_perpatient.groupby(' AnonPatientID').apply(lambda x: x.nsmallest(1, 'dateDifference')).reset_index(drop=True)
# set c is the set of studyMetadata_perpatient that has 'triple' in its description
set_c =  studyMetadata_perpatient[studyMetadata_perpatient[' StudyDescription'].str.contains('triple')]
# take the and combination of all above, drop duplicates
set_d =  studyMetadata_perpatient[studyMetadata_perpatient[' StudyDescription'].str.contains('guided')]
studyMetadata_perpatient_cleaned = pd.concat([set_c,set_a, set_b,set_d]).drop_duplicates()


print( len(studyMetadata_perpatient))
print( len(studyMetadata_perpatient_cleaned))

#plot histogram of the dateDifference
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
# three plots on one row
fig, ax = plt.subplots(1, 4, figsize=(18, 9))
ax[0].hist(studyMetadata_perpatient['dayDifference'], bins=20)
ax[0].set_title('Histogram of date difference')
ax[0].set_xlabel('Day difference')
ax[0].set_ylabel('Frequency')
#plot a bar chart of the study descriptions
studyMetadata_perpatient[' StudyDescription'] = studyMetadata_perpatient[' StudyDescription'].str.strip()
studyMetadata_perpatient[' StudyDescription'].value_counts().plot(kind='bar', ax=ax[1])
ax[1].set_title('Study Description before desciption cleaning')
ax[1].set_xlabel('Study Description')
ax[1].set_ylabel('Frequency')
#
studyMetadata_perpatient_cleaned[' StudyDescription'] = studyMetadata_perpatient_cleaned[' StudyDescription'].str.strip()
studyMetadata_perpatient_cleaned[' StudyDescription'].value_counts().plot(kind='bar', ax=ax[2])
ax[2].set_title('Study Description after desciption cleaning')
ax[2].set_xlabel('Study Description')
ax[2].set_ylabel('Frequency')
#plot a histogram of the day difference in the cleaned data
ax[3].hist(studyMetadata_perpatient_cleaned['dayDifference'], bins=20)
ax[3].set_title('Histogram of date difference in cleaned data')
ax[3].set_xlabel('Day difference')
ax[3].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

#save the cleaned data
studyMetadata_perpatient_cleaned.to_csv('/Users/mcgoug01/Downloads/Access_list_cleaned.csv',index=False)
