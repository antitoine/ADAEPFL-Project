import { Pipe, PipeTransform } from '@angular/core';
//import * as _ from 'lodash';
declare let _:any;

@Pipe({
  name: 'range'
})
export class RangePipe implements PipeTransform {

  transform(end: number, start: number = 0, step: number = 1): any {
    return _.range(start, end, step);
  }

}
