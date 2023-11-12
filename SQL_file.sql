--display all the tables in your database--
SELECT *
FROM Customer_Details cd
JOIN Account_Details ad ON cd.Customer_ID = ad.Customer_ID
JOIN Transaction_Details td ON ad.Customer_ID = td.Customer_ID
ORDER BY td.Customer_ID;

--find customers with null or missing values in any of their fields--
SELECT *
FROM Customer_Details
WHERE Customer_ID IS NULL
   OR Account_Number IS NULL
   OR Date_of_Birth IS NULL
   OR Email_Address IS NULL
   OR Credit_Scores IS NULL;


--Query Demonstrating Nominal Data--
SELECT Account_Number, Email_Address
FROM Customer_Details
LIMIT 5;

--Query Demonstrating Ordinal Data--
SELECT Account_Name, Risk_Ratings
FROM Account_Details;

--Query Demonstrating Interval Data--
SELECT Transaction_Ref, Trnx_Date, Trnx_Amount
FROM Transaction_Details
WHERE Trnx_Date BETWEEN '2023-01-01' AND '2023-06-30';

--Query Demonstrating Ratio Data--
SELECT Account_Name, Balance
FROM Account_Details
WHERE Balance > 10000
LIMIT 5;

--QUERY demonstrating duplicate values--
--a--
SELECT Account_Name, COUNT(*)
FROM Account_Details
GROUP BY Account_Name
HAVING COUNT(*) > 1;
--b--
SELECT Email_Address, COUNT(*)
FROM Customer_Details
GROUP BY Email_Address
HAVING COUNT(*) > 1;

--Diplay Inactive Accounts with big balances--
SELECT *
FROM Account_Details
WHERE Account_Status = 'Inactive' AND Balance > 100000;

--Top Ten active customers--
SELECT ad.Customer_ID, ad.Account_Name, ad.Balance, ad.Account_Status
FROM Account_Details ad
WHERE ad.Account_Status = 'Active'
ORDER BY ad.Balance DESC
LIMIT 10;

