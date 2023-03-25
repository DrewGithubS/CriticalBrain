#include <cstdint>

float f16ToF32(uint16_t fpIn)
{
  // MSB -> LSB
  // float16=1bit: sign, 5bit: exponent, 10bit: fraction
  // float32=1bit: sign, 8bit: exponent, 23bit: fraction
  // for normal exponent(1 to 0x1e): value=2**(exponent-15)*(1.fraction)
  // for denormalized exponent(0): value=2**-14*(0.fraction)

  uint32_t sign = fpIn >> 15;
  uint32_t exponent = (fpIn >> 10) & 0x1F;
  uint32_t fraction = (fpIn & 0x3FF);
  uint32_t float32Value;
  if (exponent == 0)
  {
    if (fraction == 0)
    {
      // zero
      float32Value = (sign << 31);
    }
    else
    {
      // can be represented as ordinary value in float32
      // 2 ** -14 * 0.0101
      // => 2 ** -16 * 1.0100
      // int int_exponent = -14;
      exponent = 127 - 14;
      while ((fraction & (1 << 10)) == 0)
      {
        //int_exponent--;
        exponent--;
        fraction <<= 1;
      }
      fraction &= 0x3FF;
      // int_exponent += 127;
      float32Value = (sign << 31) | (exponent << 23) | (fraction << 13);  
    }    
  }
  else if (exponent == 0x1F)
  {
    /* Inf or NaN */
    float32Value = (sign << 31) | (0xFF << 23) | (fraction << 13);
  }
  else
  {
    /* ordinary number */
    float32Value = (sign << 31) | ((exponent + (127-15)) << 23) | (fraction << 13);
  }
  
  return *((float*)&float32Value);
}