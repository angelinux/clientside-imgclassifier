var bodyElem = $("body");
var canvasUrl = $('#image_url_preview')[0].getContext('2d');
var canvasUpload = $('#image_upload_preview')[0].getContext('2d');

const MOBILENET_MODEL_PATH =
    // tslint:disable-next-line:max-line-length
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 3;

let mobilenet;


$(function(){

    $("#image_url_form").submit(function(e){
        e.preventDefault();
        bodyElem.addClass('loading');        
        console.log($("#imageURL").val());
        urlVal = $("#imageURL").val();
        //$("#image_url_preview").attr("src", urlVal);
        image = new Image();
        image.crossOrigin = "Anonymous";  // This enables CORS
        image.onload = function (event) {
            try {
                canvasUrl.drawImage(image, 0, 0, IMAGE_SIZE, IMAGE_SIZE); 
                predict(canvasUrl.canvas, $('#image_url_console')[0]);   
            } catch (e) {
                alert(e);
            }
        };
        image.src = "https://cors-anywhere.herokuapp.com/"+urlVal;
        
    });

    $('input[type="file"]').change(function(e){
        var fileName = e.target.files[0].name;
        $('.custom-file-label').html(fileName);
    });

    $('#inputGroupFileAddon').on('click', function(){
        console.log("submit clicked");
        uploadElem = $('#imageUpload')[0];
        if (!uploadElem.files[0]) {
            $('.custom-file-label').html('Please specify a file');
            return;
        }
        file = uploadElem.files[0];
        if (!file.type.match('image.*')) {
            $('.custom-file-label').html('This is not an image');
            return;
        }
        let reader = new FileReader();
        // Closure to capture the file information.
        reader.onload = e => {
        // Fill the image & call predict.
            image = new Image();
            image.onload = function(event) {
                try {
                    canvasUpload.drawImage(image, 0, 0, IMAGE_SIZE, IMAGE_SIZE); 
                    predict(canvasUpload.canvas, $('#image_upload_console')[0]);
                } catch (e) {
                    alert(e);
                }
            }
            image.src = e.target.result;
        };
      
          // Read in the image file as a data URL.
        reader.readAsDataURL(file);
    });

    startup();
});

const startup = async () => {
    status('Loading modelXXX...');
  
    mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
  
    // Warmup the model. This isn't necessary, but makes the first prediction
    // faster. Call `dispose` to release the WebGL memory allocated for the return
    // value of `predict`.
    mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
  
    status('DONE');    
};

/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
async function predict(imgElement, outElement) {
  status('Predicting...');

  // The first start time includes the time it takes to extract the image
  // from the HTML and preprocess it, in additon to the predict() call.
  const startTime1 = performance.now();
  // The second start time excludes the extraction and preprocessing and
  // includes only the predict() call.
  let startTime2;
  const logits = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    const img = tf.browser.fromPixels(imgElement).toFloat();

    const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = img.sub(offset).div(offset);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    startTime2 = performance.now();
    // Make a prediction through mobilenet.
    return mobilenet.predict(batched);
  });

  // Convert logits to probabilities and class names.
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  status(`Done in ${Math.floor(totalTime1)} ms ` +
      `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);

  // Show the classes in the DOM.
  showResults(outElement, classes);

  bodyElem.removeClass('loading');
}

/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */
async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: IMAGENET_CLASSES[topkIndices[i]],
      probability: topkValues[i]
    })
  }
  return topClassesAndProbs;
}

//
// UI
//

function showResults(outElement, classes) {
  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'container';

//   const imgContainer = document.createElement('div');
//   imgContainer.appendChild(imgElement);
//   predictionContainer.appendChild(imgContainer);

  const probsContainer = document.createElement('div');
  const row = document.createElement('div');
  row.className = 'row';
  const titleElement = document.createElement('div');
  titleElement.className = 'col';
  titleElement.innerHTML = '<h3>Predictions:</h3>';
  row.appendChild(titleElement);
  probsContainer.appendChild(row);

  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement('div');
    row.className = 'row';

    const classElement = document.createElement('div');
    classElement.className = 'col';
    classElement.innerText = classes[i].className;
    row.appendChild(classElement);

    const probsElement = document.createElement('div');
    probsElement.className = 'col';
    probsElement.innerText = classes[i].probability.toFixed(3);
    row.appendChild(probsElement);

    probsContainer.appendChild(row);
  }
  //predictionContainer.appendChild(probsContainer);
  while (outElement.firstChild) {
    outElement.removeChild(outElement.firstChild);
}
  outElement.appendChild(probsContainer);
}

// mobilenetDemo();


function status(msg) {
    console.log(msg);
}

