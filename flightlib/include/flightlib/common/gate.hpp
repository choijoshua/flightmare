#include "flightlib/common/types.hpp"

namespace flightlib {

struct Gate {
    Scalar id{0};
    Vector<3> pos{0, 0, 0};
    Vector<3> ori{0, 0, 0};
};

}