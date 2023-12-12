from flask import Flask,redirect,url_for,render_template,request

from functions import clean_up_sentence,bow,predict_class,getResponse,model,intents_json,classes,words


app = Flask(__name__)

name = []
output = []

resposed =dict()

@app.route("/",methods=['GET',"POST"])
def process ():
        hook = ""
        hook = request.form.get('name')
        out = ""

        if hook !=  None:  
                name.append(hook)
                
                A = predict_class(hook,model)
                out =  getResponse(A,intents_json)

                


                
                output.append(out)

        

        print(out)


        
        final = [list(pair) for pair in zip(name, output)]
        print(final)

       
        return render_template('index.html', name=name,Final = final,namecount=len(name))

        
        

    


# @app.route("/<name>/<age>")
# def random (name,age):
#     return f"hHello {name} age is {age} worlds"

# @app.route("/admin")
# def admin():
#     return redirect(url_for("home"))



if __name__ == "__main__":
    app.run()