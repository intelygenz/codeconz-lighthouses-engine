var palette = [
  0xff0000, 0x580aff, 0xdeff0a, 0xbe0aff, 0x0aefff, 0x51ff0a, 0xff8700,
  0x147df5, 0x0aff99, 0xffd300,
];

palette = [
  0xff0000, 0xffff00, 0x00ff00, 0x00ffff, 0x0000ff, 0xff00ff, 0xff0000,
  0xffff00, 0x00ff00, 0x00ffff,
];

palette = [
  0xff0000, 0xff8000, 0xffff00, 0x008000, 0x0000ff, 0x800080, 0xff0000,
  0xff8000, 0xffff00, 0x008000,
];

palette = [
  0xff595e, 0xff924c, 0xffca3a, 0x8ac926, 0x1982c4, 0x6a4c93, 0xff595e,
  0xff924c, 0xffca3a, 0x8ac926,
];

export const colorFor = (index) => palette[index];

export const colorToString = (color) => {
  // Convert the number to a hexadecimal string
  let hexString = color.toString(16);

  // Ensure the string is 6 characters long, adding leading zeros if necessary
  while (hexString.length < 6) {
    hexString = "0" + hexString;
  }

  // Prepend the pound sign to make it a valid CSS hex color
  return hexString;
};

export const colorToHex = (color) => {
  return "#" + colorToString(color);
};
