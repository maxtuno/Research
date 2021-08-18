"""
///////////////////////////////////////////////////////////////////////////////
//        Copyright (c) 2012-2020 Oscar Riveros. all rights reserved.        //
//                        oscar.riveros@peqnp.science                        //
//                                                                           //
//   without any restriction, Oscar Riveros reserved rights, patents and     //
//  commercialization of this knowledge or derived directly from this work.  //
///////////////////////////////////////////////////////////////////////////////

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import pickle

import gym
import numpy
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # The purpose of this model is not to be constant at score, is reach the best score possible within 100 rounds.

    env = gym.make('Boxing-ram-v4')
    model = pickle.load(open('model.pkl', 'rb'))
    neutral_point = numpy.loadtxt('neutral_point.txt')

    tim, scr = [], []

    for step in range(100):
        score = 0
        state = env.reset()
        model.fit([state], neutral_point)
        while True:
            env.render()
            state, reward, done, info = env.step(numpy.argmax(model.predict(state)))
            score += reward
            if done:
                break
        scr.append(score)
        tim.append(step)
        plt.plot(tim, scr, 'r-')
        plt.title('Dynamic Learning by Continuous Stimulus - www.peqnp.com')

        plt.savefig('score.png')
    env.close()
