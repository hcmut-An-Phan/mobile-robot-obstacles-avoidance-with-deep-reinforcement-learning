#ifndef DIFFDRIVE_ESP32__VISIBILITY_CONTROL_H_
#define DIFFDRIVE_ESP32__VISIBILITY_CONTROL_H_

// This logic was borrowed (then namespaced) from the examples on the gcc wiki:
//     https://gcc.gnu.org/wiki/Visibility

#if defined _WIN32 || defined __CYGWIN__
#ifdef __GNUC__
#define DIFFDRIVE_ESP32_EXPORT __attribute__((dllexport))
#define DIFFDRIVE_ESP32_IMPORT __attribute__((dllimport))
#else
#define DIFFDRIVE_ESP32_EXPORT __declspec(dllexport)
#define DIFFDRIVE_ESP32_IMPORT __declspec(dllimport)
#endif
#ifdef DIFFDRIVE_ESP32_BUILDING_DLL
#define DIFFDRIVE_ESP32_PUBLIC DIFFDRIVE_ESP32_EXPORT
#else
#define DIFFDRIVE_ESP32_PUBLIC DIFFDRIVE_ESP32_IMPORT
#endif
#define DIFFDRIVE_ESP32_PUBLIC_TYPE DIFFDRIVE_ESP32_PUBLIC
#define DIFFDRIVE_ESP32_LOCAL
#else
#define DIFFDRIVE_ESP32_EXPORT __attribute__((visibility("default")))
#define DIFFDRIVE_ESP32_IMPORT
#if __GNUC__ >= 4
#define DIFFDRIVE_ESP32_PUBLIC __attribute__((visibility("default")))
#define DIFFDRIVE_ESP32_LOCAL __attribute__((visibility("hidden")))
#else
#define DIFFDRIVE_ESP32_PUBLIC
#define DIFFDRIVE_ESP32_LOCAL
#endif
#define DIFFDRIVE_ESP32_PUBLIC_TYPE
#endif

#endif  // DIFFDRIVE_ESP32__VISIBILITY_CONTROL_H_
