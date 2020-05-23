#undef min
#undef max

#include <EloquentTinyML.h>

#include "text_model.h"

#define NUMBER_OF_INPUTS 8
#define NUMBER_OF_OUTPUTS 2
// in future projects you may need to tweek this value: it's a trial and error process
#define TENSOR_ARENA_SIZE 2*1024

Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml((unsigned char*)model_data);

void printPredicted(float predicted[NUMBER_OF_OUTPUTS]) {
    Serial.print("\t predicted: ");
    Serial.print(predicted[0]);
    Serial.print("\t");
    Serial.println(predicted[1]);
}

void setup() {
    while (!Serial) ; //Needed for the MKR Zero to initialise it's virtual serial port

    Serial.begin(115200);

    unsigned long start, finished, elapsed;

    delay(2000);

    Serial.println("Calculating...");

    start=millis();

    float input[8] = {2, 0, 4, 0, 0, 0, 0, 0};
    float predicted[2];
   
    ml.predict(input,predicted);

    finished=millis();

    printPredicted(predicted);

    float input2[8] = {0, 0, 0, 0, 0, 0, 0, 0};
   
    ml.predict(input2,predicted);

    printPredicted(predicted);

    float input3[8] = {2, 0, 5, 6, 0, 0, 0, 0}; 
    ml.predict(input3,predicted);
    printPredicted(predicted);

    float input4[8] =  {0,0,5,0,0,0,0,0}; 
    ml.predict(input4,predicted);
    printPredicted(predicted);

    Serial.print("\t duration: ");
    elapsed=finished-start;
    Serial.println(elapsed);
}

void loop() {
    delay(200);
}
