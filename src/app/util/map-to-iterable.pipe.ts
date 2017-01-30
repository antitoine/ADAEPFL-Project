import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'mapToIterable',
  pure: false
})
export class MapToIterablePipe implements PipeTransform {

  transform(dict: Object): {key: any, value: any}[] {
    let result = [];
    for (let key in dict) {
      if (dict.hasOwnProperty(key)) {
        result.push({key: key, value: dict[key]});
      }
    }
    return result;
  }

}
