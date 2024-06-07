/* 
   Define single-letter commands that will be sent by the PC over the serial link.
*/

#ifndef COMMANDS_H
#define COMMANDS_H

#define PING                           'p'
#define RECONNECT                      'c'
#define GET_BAUDRATE                   'b'

#define SET_VELOCITY                   'v'
#define GET_VELOCITY                   'g'

#define GO_CIRCLE                      'r'
#define LINEAR                         'l'
#define ANGULAR                        'a'

#endif //COMMANDS_H


