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

palette = [
  0xff0000, 0xff8700, 0xffd300, 0xdeff0a, 0xa1ff0a, 0x0aff99, 0x0aefff,
  0x147df5, 0x580aff, 0xbe0aff, 0xcccccc,
];

export const colorFor = (index: number): number => palette[index];

export const colorToString = (color: number): String => {
  let hexString = color.toString(16);
  while (hexString.length < 6) {
    hexString = "0" + hexString;
  }
  return hexString;
};

export const colorToHex = (color: number): String => {
  return "#" + colorToString(color);
};
