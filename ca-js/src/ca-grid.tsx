import { Component, ReactElement } from "react";
import { MultiImgCAModel } from "./multi-img-ca";

/**
 * An abstract class for managing presentation of Cellular Automata. 
 */
export abstract class GridCA {
    
    /**
     * Initialize any state for the CA. On promise resolution, 
     * calls to rgba() and other synchronous methods should succeed.
     */
    abstract init(): Promise<GridCA>;

    /**
     * Take 1 step in the CA process. This may impact results of subsequent 
     * rgba() calls.
     */
    abstract step(): void;


    /**
     * Provide an RGBA representation of the CA for presentation. RGBA values should be
     * between [0, 1]. The result should be consumable via: 
     * https://pixijs.download/release/docs/PIXI.Texture.html#fromBuffer
     */
    abstract rgba(): Float32Array;
}

export class SimpleGridCA extends GridCA {
    
    array: Float32Array;
    constructor() {
        super();
        this.array = new Float32Array(65536);
    }
    init(): Promise<GridCA> {
        this.array.fill(Math.random());
        return Promise.resolve(this);
    }
    step(): void {
        for (let i = 0; i < 65536; i += Math.floor(Math.random() * Math.floor(500))) {
            this.array[i] = Math.random();
        }
    }
    rgba(): Float32Array {
        return this.array;
    }

}

export interface GridCAViewOptions {
    maxFPS: number,
    height: number,
    width: number,
    maxSteps?: number
}