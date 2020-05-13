#undef min
#undef max

#include <EloquentTinyML.h>

#include "text_model.h"

#define NUMBER_OF_INPUTS 16
#define NUMBER_OF_OUTPUTS 1
// in future projects you may need to tweek this value: it's a trial and error process
#define TENSOR_ARENA_SIZE 2*1024

Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml((unsigned char*)model_data);

void setup() {
    Serial.begin(115200);

    float input[16] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
    float predicted = ml.predict(input);

    Serial.print("\t predicted: ");
    Serial.println(predicted);
    delay(1000);
    
}

void loop() {

}
