## 소스코드의 (변수,함수,클래스)네이밍 데이터 분석해 보기

### 개요
Java 소스코드의 네이밍 패턴을 추출하여 어떤 규칙이 있는지를 데이터로 탐색해 본다. 
<br>주요 내용은 https://brunch.co.kr/@goodvc78/12 참조 

### 구성 
#### 소스코드에서 클래스, 함수, 변수 네이밍 추출 및 네이밍 데이터 셋 생성  
01. extract-naming-words-in-java-source.ipynb
<br> 결과 데이터셋 파일이 ./resource/*.pkl로 생성됨
#### 생성된 네이밍 데이터셋 탐색 
02. naming-dataset-analysis.ipynb 

### 분석 할 소스 저장소 
* 아래 경로에 네이밍을 분석할 소스를 넣는다. 
 ./resource/source
* Elastic Search 소스를 넣고 분석할 경우 예시 
<pre>
cd ./resource/source
git clone https://github.com/elastic/elasticsearch.git
</pre> 

### 참고 자료 
* github 인기 저장소 : http://github-rank.com/star?language=Java 
* 주석 찾기 패턴 : http://blog.ostermiller.org/find-comment 
* Java Naming Convention : https://en.wikipedia.org/wiki/Naming_convention_(programming)#Java  
* 코드 하이라이트: http://colorscripter.com/ 
* 형태소 분석기 : http://www.nltk.org/ 
* Gensim의 TF-IDF:https://radimrehurek.com/gensim/models/tfidfmodel.html 
* Python Word Cloud Package : https://github.com/amueller/word_cloud  


