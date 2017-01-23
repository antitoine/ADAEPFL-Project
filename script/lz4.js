let parseArgs = require('minimist');

let fs = require('fs');
let lz4 = require('lz4');

let args = parseArgs(process.argv.slice(2));

if ( !args.hasOwnProperty('_') || args['_'].length == 0
  || (args.hasOwnProperty('h') && args['h'])
  || (
    (!args.hasOwnProperty('c') || !args['c'])
    &&
    (!args.hasOwnProperty('u') || !args['u'])
  )) {

  process.stdout.write(
    'Usage:\n' +
    'lz4.js inputFile [-c|-u] [options]\n' +
    '\n' +
    'Options:\n' +
    ' -h                 \t Display this help\n' +
    ' -o outputFile      \t Redirect output in a file\n' +
    ' -c                 \t Compress the input file\n' +
    ' -u                 \t Uncompress the input file\n'
  );
  return;
}

let input = fs.readFileSync(args['_'][0])

let output;
if (args.hasOwnProperty('c') && args['c']) {
  process.stdout.write('Compressing ... ');
  output = lz4.encode(input);
} else {
  process.stdout.write('Uncompressing ... ');
  output = lz4.decode(input);
}
process.stdout.write(' Done.\n');

if (args.hasOwnProperty('o') && args['o']) {
  process.stdout.write('Write output file ... ');
  fs.writeFileSync(args['o'], output);
  process.stdout.write(' Done (' + args['o'] + ').\n');
} else {
  process.stdout.write('Result:\n' + output + '\n');
}
