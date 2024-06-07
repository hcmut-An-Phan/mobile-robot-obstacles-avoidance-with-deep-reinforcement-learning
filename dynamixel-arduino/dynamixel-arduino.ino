#include <Dynamixel2Arduino.h>
#include "commands.h"
#include "turtlebot3_waffle.h"

#define DXL_SERIAL         Serial2
#define DEBUG_SERIAL       Serial
#define DEBUG_BAUDRATE     57600
#define DXL_BAURATE        1000000


/* functions prototype */
void syncSetVelocity(float right_velocity, float left_velocity);
void syncReadVelocity(void);

void FindServos(void);
void tryConnect(void);

void resetCommand(void); 
void runCommand(void);

/* Variables for sync read write */
const uint8_t BROADCAST_ID = 254;
const uint8_t DXL_ID_CNT = 2;
const uint8_t DXL_ID_LIST[DXL_ID_CNT] = {1, 2};
const uint16_t user_pkt_buf_cap = 128;
uint8_t user_pkt_buf[user_pkt_buf_cap];

const uint16_t SR_START_ADDR = 126; // Starting Data Addr, present velocity
const uint16_t SR_ADDR_LEN = 10; // Data Length 4 byte 

const uint16_t SW_START_ADDR = 104; 
const uint16_t SW_ADDR_LEN = 4;

// struct read data 
typedef struct sr_data{
  int16_t present_current;
  int32_t present_velocity;
  int32_t present_position;
} __attribute__((packed)) sr_data_t;

// struct write data, data type must be uint32_t 
typedef struct sw_data{
  int32_t goal_velocity;
} __attribute__((packed)) sw_data_t;


sr_data_t sr_data[DXL_ID_CNT];
DYNAMIXEL::InfoSyncReadInst_t sr_infos;
DYNAMIXEL::XELInfoSyncRead_t info_xels_sr[DXL_ID_CNT];

sw_data_t sw_data[DXL_ID_CNT]; 
DYNAMIXEL::InfoSyncWriteInst_t sw_infos; 
DYNAMIXEL::XELInfoSyncWrite_t info_xels_sw[DXL_ID_CNT];

/* DXL Variable initialization */
const int DXL_DIR_PIN = 2; // DYNAMIXEL Shield DIR PIN
const uint8_t LEFT_MOTOR_ID = 1;
const uint8_t RIGHT_MOTOR_ID = 2;
const float DXL_PROTOCOL_VERSION = 2.0;

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
DYNAMIXEL::InfoFromPing_t ping_info[32];

//This namespace is required to use Control table item names
using namespace ControlTableItem;

/* Variables to handle message from pc*/
// A pair of varibles to help parse serial commands
int arg = 0;
int _index = 0;

// Variable to hold an input character
char chr;

// Variable to hold the current single-character command
char cmd;

// Character arrays to hold the first and second arguments
char argv1[16];
char argv2[16];

// The arguments converted to float
float arg1; 
float arg2; 

void setup() 
{
  uint8_t i;

  // Use UART port of DYNAMIXEL Shield to debug.
  DEBUG_SERIAL.begin(DEBUG_BAUDRATE);
  
  // Set Port baudrate to 1000000bps. This has to match with DYNAMIXEL baudrate.
  dxl.begin(DXL_BAURATE);
  
  // Set Port Protocol Version. This has to match with DYNAMIXEL protocol version.
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);
  
  // Get DYNAMIXEL information
  dxl.ping(RIGHT_MOTOR_ID);
  dxl.ping(LEFT_MOTOR_ID);

  // Turn off torque when configuring items in EEPROM area
  for(i=0; i<DXL_ID_CNT; i++){
    dxl.torqueOff(DXL_ID_LIST[i]);
    dxl.setOperatingMode(DXL_ID_LIST[i], OP_VELOCITY);
  }
  dxl.torqueOn(BROADCAST_ID);

  /* Setup for sync read write */
  // Fill the members of structure to syncRead using external user packet buffer
  sr_infos.packet.p_buf = user_pkt_buf;
  sr_infos.packet.buf_capacity = user_pkt_buf_cap;
  sr_infos.packet.is_completed = false;
  sr_infos.addr = SR_START_ADDR;
  sr_infos.addr_length = SR_ADDR_LEN;
  sr_infos.p_xels = info_xels_sr;
  sr_infos.xel_count = 0;  

  for(i=0; i<DXL_ID_CNT; i++){
    info_xels_sr[i].id = DXL_ID_LIST[i];
    info_xels_sr[i].p_recv_buf = (uint8_t*)&sr_data[i];
    sr_infos.xel_count++;
  }
  sr_infos.is_info_changed = true;

  // Fill the members of structure to syncWrite using internal packet buffer
  sw_infos.packet.p_buf = nullptr;
  sw_infos.packet.is_completed = false;
  sw_infos.addr = SW_START_ADDR;
  sw_infos.addr_length = SW_ADDR_LEN;
  sw_infos.p_xels = info_xels_sw; // pxel pointer 
  sw_infos.xel_count = 0;

  // make data to write
  sw_data[0].goal_velocity = 0;
  sw_data[1].goal_velocity = 0;
  
  for(i=0; i<DXL_ID_CNT; i++){
    info_xels_sw[i].id = DXL_ID_LIST[i];
    info_xels_sw[i].p_data = (uint8_t*)&sw_data[i].goal_velocity;
    sw_infos.xel_count++;
  }
   sw_infos.is_info_changed = true; // update sending info after change velocity
}

void loop() {
  while (DEBUG_SERIAL.available() > 0) 
  {
    // Read the next character
    chr = DEBUG_SERIAL.read();
    
    // Terminate a command with a Enter
    if (chr == 13) 
    {
      if (arg == 1) argv1[_index] = NULL;
      else if (arg == 2) argv2[_index] = NULL;
      runCommand();
      resetCommand();
    }
    
    // Use spaces to delimit parts of the command
    else if (chr == ' ') 
    {
      // Step through the arguments
      if (arg == 0) arg = 1;
      else if (arg == 1)  
      {
        argv1[_index] = NULL;
        arg = 2;
        _index = 0;
      }
      continue;
    }
    else 
    {
      if (arg == 0) 
      {
        // The first arg is the single-letter command
        cmd = chr;
      }
      else if (arg == 1) 
      {
        // Subsequent arguments can be more than one character
        argv1[_index] = chr;
        _index++;
      }
      else if (arg == 2) 
      {
        argv2[_index] = chr;
        _index++;
      }
    }
  }
 }

/* Control velocity of 2 motors*/
void syncSetVelocity(float left_velocity, float right_velocity )
{
  // convert rpm float data to raw int32_t data
  sw_data[0].goal_velocity = (int32_t)round(left_velocity/0.229);
  sw_data[1].goal_velocity = (int32_t)round(right_velocity/0.229);
  sw_infos.is_info_changed = true;
  dxl.syncWrite(&sw_infos);
}

/* Read state of 2 motors */
void syncReadVelocity(void)
{
  uint8_t i, recv_cnt;
  recv_cnt = dxl.syncRead(&sr_infos);
  if(recv_cnt > 0){
    DEBUG_SERIAL.print("[SyncRead] Success, Received ID Count: ");
    DEBUG_SERIAL.println(recv_cnt);
    for(i=0; i<recv_cnt; i++){
      DEBUG_SERIAL.print("  ID: ");DEBUG_SERIAL.print(sr_infos.p_xels[i].id);
      DEBUG_SERIAL.print(", Error: ");DEBUG_SERIAL.println(sr_infos.p_xels[i].error);
      DEBUG_SERIAL.print("\t Present Current: ");DEBUG_SERIAL.println(sr_data[i].present_current);
      DEBUG_SERIAL.print("\t Present Velocity: ");DEBUG_SERIAL.println(sr_data[i].present_velocity);
      DEBUG_SERIAL.print("\t Present Position: ");DEBUG_SERIAL.println(sr_data[i].present_position);
    }
  }else{
    DEBUG_SERIAL.print("[SyncRead] Fail, Lib error code: ");
    DEBUG_SERIAL.println(dxl.getLastLibErrCode());
  }
}

void FindServos(void) 
{
  DEBUG_SERIAL.println("Try Protocol 2 - broadcast ping: ");
  DEBUG_SERIAL.flush(); // flush it as ping may take awhile... 
      
  if (uint8_t count_pinged = dxl.ping(DXL_BROADCAST_ID, ping_info, sizeof(ping_info)/sizeof(ping_info[0]))) 
  {
    DEBUG_SERIAL.print("Detected Dynamixel : \n");
    for (int i = 0; i < count_pinged; i++)
    {
      DEBUG_SERIAL.print("ID: ");
      DEBUG_SERIAL.print(ping_info[i].id, DEC);
      DEBUG_SERIAL.print(", Model:");
      DEBUG_SERIAL.print(ping_info[i].model_number);
      DEBUG_SERIAL.print(", Ver:");
      DEBUG_SERIAL.println(ping_info[i].firmware_version, DEC);
    }
  }
  else
  {
    DEBUG_SERIAL.print("Broadcast returned no items : ");
    DEBUG_SERIAL.print("Error code: ");
    DEBUG_SERIAL.println(dxl.getLastLibErrCode());
  }
}

void tryConnect(void)
{
  // Set Port baudrate to 1000000bps. This has to match with DYNAMIXEL baudrate.
  dxl.begin(DXL_BAURATE);

  if (dxl.readControlTableItem(BAUD_RATE, RIGHT_MOTOR_ID) == 3)
    DEBUG_SERIAL.println("reconnect right motor successed");
  else 
    DEBUG_SERIAL.println("reconnect right motor failed");

  if (dxl.readControlTableItem(BAUD_RATE, LEFT_MOTOR_ID) == 3)
    DEBUG_SERIAL.println("reconnect left motor successed");
  else 
    DEBUG_SERIAL.println("reconnect left motor failed");

  // Set Port Protocol Version. This has to match with DYNAMIXEL protocol version.
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  //Turn off torque when configuring items in EEPROM area
  dxl.torqueOff(RIGHT_MOTOR_ID);
  dxl.torqueOff(LEFT_MOTOR_ID);

  dxl.setOperatingMode(RIGHT_MOTOR_ID, OP_VELOCITY);
  dxl.setOperatingMode(LEFT_MOTOR_ID, OP_VELOCITY);

  dxl.torqueOn(RIGHT_MOTOR_ID);
  dxl.torqueOn(LEFT_MOTOR_ID);
}

/* Clear the current command parameters */
void resetCommand(void) 
{
  cmd = NULL;
  memset(argv1, 0, sizeof(argv1));
  memset(argv2, 0, sizeof(argv2));
  arg1 = 0;
  arg2 = 0;
  arg = 0;
  _index = 0;
}

/* Run a command.  Commands are defined in commands.h */
void runCommand(void) 
{
  arg1 = atof(argv1);
  arg2 = atof(argv2);
  
  switch(cmd) {
    case RECONNECT:
      tryConnect();
    break;

    case GET_BAUDRATE:
      DEBUG_SERIAL.println(DEBUG_BAUDRATE);
    break;

    case PING:
      FindServos();
    break;

    case SET_VELOCITY:
      VelocityControl(arg1, arg2);
    break;

   case GET_VELOCITY:
      syncReadVelocity();
    break;

    default:
      DEBUG_SERIAL.println("Invalid Command");
    break;
  }
}
