import * as tf from '@tensorflow/tfjs';
import { LayersModel, Rank, Tensor, Tensor2D, Tensor3D, Tensor4D, Tensor5D} from "@tensorflow/tfjs";


export const DEFAULT_CELL_FIRE_RATE: number = 0.5;
const DEFAULT_BOARD_CHANNEL_N = 16;
const DEFAULT_IMG_CHANNEL_N = 72;

export class MultiImgCAModel {

    readonly imgModel: LayersModel;
    readonly updateModel: LayersModel;
    readonly boardChannelN: number;
    readonly imgChannelN: number;
    readonly postImgChannelN: number;
    readonly perceptionKernel: Tensor4D;
    readonly fireRate: number;
    readonly applyLivingMask: boolean = true;
    postImgFilters: Tensor4D | undefined;
    
    /**
     * 
     * @param imgModel 
     * @param updateModel 
     * @param boardChannelN 
     * @param fireRate 
     * @param imgChannelN 
     */
    constructor(imgModel: LayersModel, updateModel: LayersModel, boardChannelN: number = DEFAULT_BOARD_CHANNEL_N, 
        fireRate: number = DEFAULT_CELL_FIRE_RATE, imgChannelN: number = DEFAULT_IMG_CHANNEL_N) {
        this.imgModel = imgModel;
        this.updateModel = updateModel;

        // channel counts, for model component construction and tracking
        this.boardChannelN = boardChannelN;
        this.imgChannelN = imgChannelN;
        this.postImgChannelN = imgChannelN + 4;

        // construct perception kernel: not a saved model piece
        this.perceptionKernel = this.constructPerceptionKernel(this.postImgChannelN);
        this.fireRate = fireRate
    }
    
    /**
     * 
     * @param channelN 
     */
    constructPerceptionKernel(channelN: number): Tensor4D {
        const identity: Tensor2D = tf.tensor2d([[0,0,0], [0,1,0], [0,0,0]]);
        const sobX: Tensor2D = tf.div(tf.outerProduct([1, 2, 1], [-1, 0, 1]), tf.scalar(8.0));
        const sobY: Tensor2D = tf.transpose(sobX);
        // 3 x (3x3) + new dimmention = (3,3,1,3)
        const stacked: Tensor4D = tf.expandDims(tf.stack([identity, sobX, sobY], -1), 2);
        return tf.tile(stacked, [1, 1, channelN, 1]);
    }

    /**
     * 
     * @param x 
     * @param images 
     */
    call(x: Tensor4D, images: Tensor4D, fireRate?: number) {
        // calculate a living mask from step n
        const preLifeMask: Tensor4D = this.getLivingMask(x);

        // process the images into filters, inserting two 'empty' dimensions to allow each element in filters to be
        // 1x1 conv2d filter (1, 1, c_in, c_out) on the board
        if (!this.postImgFilters) {
            const postImage: Tensor3D = this.imgModel.apply(images) as Tensor3D;
            this.postImgFilters = postImage.expandDims(0) as Tensor4D; // TODO: nicer way to expand twice?
        }

        // compute a 1x1 batch-wise convolution of our board and our image-based filters: this is how the img model
        // communicates to the update model -- TODO: can we do this with frozen weights from an auto-encoder?
        const boardOutputOnly: Tensor4D = x.slice([0,0,0,0], [-1,-1,-1, 4])  // only (r, g, b, alpha)
        const boardHiddenOnly: Tensor4D = x.slice([0,0,0,4], [-1,-1,-1, -1]) // rest

        // the image aware board is a batch-wise convolution of 1) our image representation & 2) our current hidden state
        const imgAwareBoard: Tensor4D = tf.concat([
            boardOutputOnly,
            batchWiseConvolution(boardHiddenOnly, this.postImgFilters as Tensor4D)
        ], -1) as Tensor4D;
        const perceived: Tensor4D = this.perceive(imgAwareBoard);

        const dx: Tensor4D = (this.updateModel.call(perceived, {}) as Tensor4D[])[0];
        if (!fireRate) {
            fireRate = this.fireRate
        }
        const maskShape: [number, number, number, number] = [x.shape[0], x.shape[1], x.shape[2], 1]
        const updateMask: Tensor4D = tf.lessEqual(tf.randomUniform(maskShape), fireRate)
        x = x.add(tf.mul(dx, tf.cast(updateMask, "float32")));
        // calculate another from step n + 1
        const postLifeMask: Tensor4D = this.getLivingMask(x);
        // to be alive, you must be alive in both the current & next board
        const livingMask: Tensor4D = preLifeMask.logicalAnd(postLifeMask);
        if (this.applyLivingMask) {
            x = x.mul(tf.cast(livingMask, "float32"));
        }
        return x;
       // whats left after call? How does one do an episode? Maybe we need an inference class similar to trainer
    }

    /**
     * 
     * @param x 
     */
    perceive(x: Tensor4D): Tensor4D {
        return tf.depthwiseConv2d(x, this.perceptionKernel, 1, 'same');
    }

    /**
     * 
     * @param x 
     */
    getLivingMask(x: Tensor4D): Tensor4D {
        const alpha: Tensor4D = x.slice([0, 0, 0, 3], [-1, -1, -1, 1]);
        return tf.maxPool(alpha, 3, 1, 'same').asType("bool");
    }

    /**
     * 
     * @param basePath 
     */
    static async loadModel(basePath: string): Promise<MultiImgCAModel> {
        const imgModel: LayersModel = await tf.loadLayersModel(basePath + '/img_model/model.json');
        const updateModel: LayersModel = await tf.loadLayersModel(basePath + '/update_model/model.json');
        console.log('up', updateModel);
        console.log('img', imgModel);
        return new MultiImgCAModel(imgModel, updateModel);
    }
}

/**
 * 
 * @param inputs 
 * @param filters 
 */
// TODO move
export function batchWiseConvolution(inputs: Tensor4D, filters: Tensor4D) {
    const singleConv: (x: Tensor4D, kernel: Tensor4D) => Tensor = (x, kernel) => tf.conv2d(x, kernel, 1, 'valid');
    // tf.squeeze(tf.map_fn(single_conv, (tf.expand_dims(inputs, 1), filters), dtype=dtype), axis=1)
    const convolved: Tensor[] = [];
    for (let i: number = 0; i < inputs.shape[0]; i++) {
        const t: Tensor4D = inputs.gather(i).expandDims(0);
        const f: Tensor4D = filters.gather(i).expandDims(0);
        convolved.push(singleConv(t, f));
    }
    return tf.squeeze(tf.stack(convolved), [0])
}