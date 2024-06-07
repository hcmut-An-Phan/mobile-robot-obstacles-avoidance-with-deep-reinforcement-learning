#include "mpu6050/mpu6050_lib.h"

MPU6050 device(0x68);

int main()
{
    float roll, pitch, yaw;                 // Angle values
    float gx, gy, gz;                       // Gyro values 
    float ax, ay, az; 
	sleep(1); // Wait for the MPU6050 to stabilize

    while (true)
    {
        device.getAngle(0, &roll);
        device.getAngle(1, &pitch);
        device.getAngle(2, &yaw);

        device.getGyro(&gx, &gy, &gz);
        device.getAccel(&ax, &ay, &az);
        
        std::cout << "Gyroscope:\n";
        std::cout << "------------------\n";
        std::cout << "X: " << gx << "\n";
        std::cout << "Y: " << gy << "\n";
        std::cout << "Z: " << gz << "\n\n";

        std::cout << "Accelerometer:\n";
        std::cout << "----------------------\n";
        std::cout << "X: " << ax << "\n";
        std::cout << "Y: " << ay << "\n";
        std::cout << "Z: " << az << "\n\n";

        // std::cout << "roll: " << roll << std::endl;
        // std::cout << "pitch: " << pitch << std::endl;
        // std::cout << "yaw: " << yaw << std::endl << std::endl;

        sleep(1);
    }
    
}