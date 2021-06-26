# -*- coding: utf-8 -*-
import os
import json


def getFileList(workpath, suffix):
	fileList = []
	if os.path.exists(workpath):
		for root, subdirs, files in os.walk(workpath):
			for onefile in files:
				if onefile.endswith(suffix):
					filename = os.path.splitext(onefile)[0]
					filepath = os.path.join(root, onefile)
					fileList.append(filepath)
	return fileList

	
def __getValue(key, jsonDict, valueList,skipKey,skipValue):
	if not isinstance(jsonDict, dict):
		return jsonDict + "is not dict"
	elif key in jsonDict.keys():
		valueList.append(jsonDict[key])
	else:
		for value in jsonDict.values():
			if isinstance(value, dict):
				if skipKey in value.keys() and value[skipKey] == skipValue:
					continue
				else:
					__getValue(key, value, valueList,skipKey,skipValue)
			elif isinstance(value, (list, tuple)):
				__searchValue(key, value, valueList,skipKey,skipValue)
	return valueList


def __searchValue(key,jsonDict, valueList,skipKey,skipValue):
	for value in jsonDict:
		if isinstance(value, (list, tuple)):
			__searchValue(key, value, valueList,skipKey,skipValue)
		elif isinstance(value, dict):
			if skipKey in value.keys() and value[skipKey] == skipValue:
				continue
			else:
				__getValue(key, value, valueList,skipKey,skipValue)			
	
	

def processJson(jsonFile,key,encoding="utf8",userParam=None,skipKey=None,skipValue=None):
	dictList =[]
	try: 
		with open(jsonFile, "r",encoding=encoding) as file:
			tmp = json.load(file)
			__getValue(key, tmp, dictList,skipKey,skipValue)
	except FileNotFoundError:
		print('hahah')
	else:	
		if dictList == []:
			return userParam
		else:
			print(dictList)
			print(len(dictList))
			return dictList



if __name__=="__main__":
	processJson("./123.json","content",skipKey="type",skipValue='text')
