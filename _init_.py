from flask import Flask,request,render_template
import numpy as np
import pickle
 
app = Flask(__name__)

@app.route('/')
def homepage():

    title = "PYTHON MACHINE LEARNING"
    return render_template("ml.html")

@app.route('/about')
def aboutpage():

    title = "About this site"
    paragraph = ["blah blah blah memememememmeme blah blah memememe"]

    pageType = 'about'

    return render_template("about.html", title=title, paragraph=paragraph, pageType=pageType)




@app.route('/result',methods=['POST','GET'])
def get_delay():
    if request.method=='POST':
        result=request.form
        gdp = float(result['gdp'])
        rural = float(result['rural'])
        psb = float(result['psb'])
        reserves = float(result['reserves'])
        fpi = float(result['fpi'])
        agri = float(result['agri'])
        ex = float(result['ex'])
        oda = float(result['oda'])
        co2 = float(result['co2'])
        cap = float(result['cap'])
        frames = [gdp,rural,psb,reserves,fpi,agri,ex,oda,co2,cap]
        frames=np.asmatrix(frames,dtype=float)
        pkl_file = open('logmodel.pkl', 'rb')
        logmodel = pickle.load(pkl_file)
        prediction1 = logmodel.predict(frames)
        prediction1=prediction1*36.23-7.63     
    return render_template('result.html',prediction=prediction1)

@app.route('/contact',methods=['POST','GET'])
def contact():

    title = "PYTHON MACHINE LEARNING"
    if request.method=='POST':
        result=request.form
        name=result['name']
    
    return render_template("contact.html",name=name)

@app.route('/graph')
def graph_Example(chartID = 'chart_ID', chart_type = 'line', chart_height = 500):
    pkl = open('pred', 'rb')
    pred= pickle.load(pkl)
    pkl1 = open('ytest', 'rb')
    ytest= pickle.load(pkl1) 
    subtitleText = 'test'
#topPairs, bottomPairs = datafunctions.twoPaneGraphData('btceHistory',1, 3, 4)
    #dataSet = [[1408395614.0, 430.2], [1408395614.0, 431.13], [1408395617.0, 431.354], [1408395623.0, 432.349], [1408395623.0, 432.017], [1408395640.0, 430.195], [1408395640.0, 430.913], [1408395640.0, 430.913], [1408395647.0, 430.211], [1408395647.0, 430.297], [1408395647.0, 430.913], [1408395648.0, 432.996], [1408395648.0, 432.996], [1408395648.0, 432.349], [1408395654.0, 431.0]]
    #dataSet = np.vstack((pred, ytest))
    combi =list(zip((pred,ytest)))
    pageType = 'graph'
    chart = {"renderTo": chartID, "type": chart_type, "height": chart_height, "zoomType":'x'}
    series = [{"name": 'Label1', "data": combi}]
    graphtitle = {"text": 'My Title'}
    xAxis = {"type":"datetime"}
    yAxis = {"title": {"text": 'yAxis Label'}}
    return render_template('graph.html',pageType=pageType,subtitleText=subtitleText, chartID=chartID, chart=chart, series=series, graphtitle=graphtitle, xAxis=xAxis, yAxis=yAxis)




if __name__ == "__main__":
   app.debug = True
   app.run()
