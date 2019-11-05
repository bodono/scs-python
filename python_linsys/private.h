#ifndef PRIV_H_GUARD
#define PRIV_H_GUARD

#ifdef __cplusplus
extern "C" {
#endif

#include "amatrix.h"
#include "glbopts.h"
#include "linalg.h"
#include "scs.h"

struct SCS_LIN_SYS_WORK {
  scs_float total_solve_time;
};

#ifdef __cplusplus
}
#endif
#endif
