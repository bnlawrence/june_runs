#include <stddef.h>

/*
  Check to see what format specifier a string contains,
  if any. Returns index of the character at the end of the
  format string or 0 if there isn't one.
*/
size_t identify_format(char *str, size_t len)
{
  size_t i = 0;

  /* Advance to first '%' sign */
  while((i < len-1) && (str[i] != '%'))
    i += 1;
  
  /* Return 0 if we didn't find a % or we're at the end */
  if((str[i] != '%') || (i >= len-1))
    return 0;
  
  /* Advance to type character/flags etc */
  i += 1;
  if(i >= len)return 0;

  /* Skip over any flags */
  while((i < len-1) && 
	((str[i]=='+') ||
	 (str[i]=='-') ||
	 (str[i]==' ') ||
	 (str[i]=='#') ||
	 (str[i]=='0')))
    i += 1;
  
  /* Skip width */
  while((i < len-1) && 
	(str[i] >= '0') && 
	(str[i] <= '9'))
    i += 1;

  /* Skip precision */
  if(str[i] == '.')
    {
      /* Skip over dot */
      i+= 1;
      if(i>=len-1)
	return 0;
      
      /* Skip any digits */
      while((i < len-1) && 
	    (str[i] >= '0') && 
	    (str[i] <= '9'))
	i += 1;
    }

  /* Skip modifier */
  while((i < len-1) && 
	((str[i]=='h') ||
	 (str[i]=='l') ||
	 (str[i]=='L')))
    i += 1;

  /* Should now have type character */
  switch (str[i])
    {
    case 'd':
    case 'i':
    case 'f':
    case 'e':
    case 'E':
    case 'g':
    case 'G':
    case 's':
      return i;
    break;
    default:
      return 0;
      break;
    }
}
