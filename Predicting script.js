let mobilenet;
let model;

const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
let isPredicting = false;



async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function loadmodel() {
  const model = await tf.loadLayersModel('my_model.json');
  return model;
}


async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "One";
			break;
		case 1:
			predictionText = "Two";
			break;
		case 2:
			predictionText = "Three";
			break;
		case 3:
			predictionText = "Four";
			break;
        case 4:
			predictionText = "Five";
			break;
            
	}
	document.getElementById("prediction").innerText = predictionText;
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}




function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}





async function init(){
	await webcam.setup();
	mobilenet = await loadMobilenet();
    model = await loadmodel();
    
	tf.tidy(() => mobilenet.predict(webcam.capture()));
		
}


init();